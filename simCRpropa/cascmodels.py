import numpy as np
import copy
import logging
from gammapy.modeling.models import SpatialModel, SpectralModel
from gammapy.modeling.models import Parameter, Parameters
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.datasets.spectrum import SpectrumDatasetOnOff
from gammapy.datasets import Datasets
from scipy.interpolate import interp1d
from astropy import units as u
from collections.abc import Iterable


class DiffuseCascadeCube(SpatialModel):
    """Cube sky map model for electromagnetic cascades.

    Parameters
    ----------
    cascmap : `~simCRpropa.cascmaps.CascMap`
        Cascade Map template
    spectral_model : `SpectralModel`
        Spectral model of the injected particles.
    rotation: float
        Rotation angle of the cascade template
    tmax: float
        Maximum delay time in years allowed for the cascade.

    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.
    """

    tag = "DiffuseCascadeCube"
    rotation = Parameter("rotation", 0., unit="deg", frozen=True)
    tmax = Parameter("tmax", 1e7, unit="yr", frozen=True)
    bias = Parameter("bias", 0., unit=u.dimensionless_unscaled, frozen=True, min=-0.2, max=0.2)

    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
            self,
            cascmap,
            spectral_model,
            rotation=rotation.quantity,
            tmax=tmax.quantity,
            interp_kwargs=None,
            apply_irf=None,
            name=None,
            energy_unit_spectral_model="TeV",
    ):

        self.cascmap = cascmap
        self._spectral_model = spectral_model
        self._name = name
        self._inj_spec = None
        self._energy_unit_spectral_model = energy_unit_spectral_model
        self._set_injection_spectrum()

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

        self._cached_value = None
        self._cached_weights = None
        self._cached_coordinates = (None, None, None)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        super().__init__(tmax=tmax, rotation=rotation)

    @property
    def name(self):
        return self._name

    def _set_injection_spectrum(self):
        def inj_spec(energy, **kwargs):
            if isinstance(energy, u.Quantity):
                x = energy.to(self._energy_unit_spectral_model)
            else:
                x = energy * u.Unit(self._energy_unit_spectral_model)
            return self._spectral_model.evaluate(x, **kwargs)

        self._inj_spec = inj_spec

    def _interpolate(self, lon, lat, energy):
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
            "energy_true": energy,
        }
        return self.cascmap.add_primary_to_casc().interp_by_coord(coord, **self._interp_kwargs)

    def evaluate(self, lon, lat, energy, **kwargs):
        """Evaluate model at given coordinates"""

        rotation = kwargs.pop("rotation")
        tmax = kwargs.pop("tmax")
        smooth = kwargs.pop("smooth", True)
        bias = kwargs.pop("bias")

        # change max delay time
        if not tmax == self.cascmap.tmax:
            self.cascmap.tmax = tmax

        # change rotation angle
        # and apply rotation
        if not rotation == self.cascmap.angle:
            self.cascmap.angle = rotation

        # change spectral weights
        self.cascmap.apply_spectral_weights(injspec=self._inj_spec, smooth=smooth, **kwargs)

        is_cached_coord = [
            _ is coord for _, coord in zip((lon, lat, energy), self._cached_coordinates)
        ]

        # reset cache
        if not np.all(is_cached_coord):
            self._cached_value = None

        if self._cached_weights is not None and \
                not np.all(np.equal(self.cascmap.weights, self._cached_weights)):
            self._cached_weights = None

        if self._cached_value is None or self._cached_weights is None:
            #self._cached_coordinates = (lon, lat, energy * (1. + bias))
            #self._cached_value = self._interpolate(lon, lat, energy)
            self._cached_coordinates = (lon, lat, energy)
            self._cached_value = self._interpolate(lon, lat, energy * (1. + bias))
            self._cached_weights = self.cascmap.weights

        return u.Quantity(self._cached_value, self.cascmap.casc_obs.unit, copy=False)

    def copy(self):
        """A shallow copy"""
        new = copy.copy(self)
        return new

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.cascmap.casc_obs.geom.center_skydir

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return np.max(self.cascmap.casc_obs.geom.width) / 2.0

    @property
    def parameters(self):
        return (
                Parameters([self.rotation, self.tmax])
                + self._spectral_model.parameters
        )

    def to_dict(self):
        data = super().to_dict()
        data["name"] = self.name
        data["type"] = data.pop("type")
        data["spectral_model"] = self._spectral_model.to_dict()
        data["parameters"] = Parameters([self.rotation, self.tmax]).to_dict()

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        return data

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    # TODO covariance handling


class CascadeSpectralModel(SpectralModel):
    """Spectral model for electromagnetic cascades.

    Parameters
    ----------
    cascmap : `~simCRpropa.cascmaps.CascMap`
        Cascade Map template

    spectral_model : `SpectralModel`
        Spectral model of the injected particles.

    ebl: `EBLAbsorptionNormSpectralModel`
        EBL model to be used

    region: region object
        region in which cascade is summed up

    rotation: float
        Rotation angle of the cascade template

    tmax: float
        Maximum delay time in years allowed for the cascade.

    bias: float
        energy bias for the TeV energy spectrum

    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.

    add_primary: bool
        if true, add the observed flux from the point source in evaluate function
    """

    tag = "CascadeSpectralModel"
    rotation = Parameter("rotation", 0., unit="deg", frozen=True)
    tmax = Parameter("tmax", 1e7, unit="yr", frozen=True)
    bias = Parameter("bias", 0., unit=u.dimensionless_unscaled, frozen=True, min=-0.2, max=0.2)

    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
            self,
            cascmap,
            spectral_model,
            ebl,
            region,
            rotation=rotation.quantity,
            tmax=tmax.quantity,
            bias=bias.quantity,
            interp_kwargs=None,
            apply_irf=None,
            name=None,
            add_primary=True,
            energy_unit_spectral_model="TeV",
            use_gammapy_interp=False
    ):

        self.cascmap = cascmap
        self._intrinsic_spectral_model = spectral_model
        self._ebl = ebl
        self._name = name
        self._inj_spec = None
        self._energy_unit_spectral_model = energy_unit_spectral_model
        self._set_injection_spectrum()
        self.region = region
        self.add_primary = add_primary

        # get central sky coordinates
        self._lon, self._lat = cascmap.casc_obs.geom.center_coord[:-1]

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        #interp_kwargs.setdefault("interp", "linear")
        #interp_kwargs.setdefault("fill_value", 0)
        interp_kwargs.setdefault("interp", 1)
        self._interp_kwargs = interp_kwargs

        self._cached_value = None
        self._cached_weights = None
        self._cached_coordinates = (None, None, None)

        self.use_gammapy_interp = use_gammapy_interp

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        super().__init__(tmax=tmax, rotation=rotation, bias=bias)

    @property
    def name(self):
        return self._name

    @property
    def ebl(self):
        return self._ebl

    @property
    def intrinsic_spectral_model(self):
        return self._intrinsic_spectral_model

    def _set_injection_spectrum(self):
        def inj_spec(energy, **kwargs):
            if isinstance(energy, u.Quantity):
                x = energy.to(self._energy_unit_spectral_model)
            else:
                x = energy * u.Unit(self._energy_unit_spectral_model)
            return self._intrinsic_spectral_model.evaluate(x, **kwargs)

        self._inj_spec = inj_spec

    def _interpolate(self, energy):

        spec = self.cascmap.get_obs_spectrum(self.region)

        if self.use_gammapy_interp:
            coord = {
                "lon": self._lon,
                "lat": self._lat,
                "energy_true": energy,
            }
            result = spec.interp_by_coord(coord, **self._interp_kwargs) * spec.unit

        else:
            data = spec.data[:, 0, 0]
            data[data <= 0] = 1e-40
            x = spec.geom.axes['energy_true'].center
            interp = interp1d(np.log(x.value),
                              np.log(data),
                              fill_value='extrapolate', kind='linear'
                              )
            result = np.exp(interp(np.log(energy.to(x.unit).value))) * spec.unit

        return result

    def evaluate(self, energy, **kwargs):
        """Evaluate model at given coordinates"""

        rotation = kwargs.pop("rotation")
        tmax = kwargs.pop("tmax")
        smooth = kwargs.pop("smooth", True)
        bias = kwargs.pop("bias")

        # change max delay time
        if not tmax == self.cascmap.tmax:
            self.cascmap.tmax = tmax

        # change rotation angle
        # and apply rotation
        if not rotation == self.cascmap.angle:
            self.cascmap.angle = rotation

        # calculate flux from observed point source specturm
        # first the ebl contribution
        # and remove parameters from kwargs that belong to the EBL model
        kwargs_ebl = {}
        for k in self._ebl.parameters.names:
            kwargs_ebl[k] = kwargs.pop(k)

        result = self._ebl.evaluate(energy * (1. + bias), **kwargs_ebl)
        result *= self._intrinsic_spectral_model.evaluate(energy * (1. + bias), **kwargs)

        # change spectral weights
        self.cascmap.apply_spectral_weights(injspec=self._inj_spec,
                                            smooth=smooth,
                                            force_recompute=True,
                                            **kwargs)

        is_cached_coord = [
            _ is coord for _, coord in zip(energy, self._cached_coordinates)
        ]

        # reset cache
        if not np.all(is_cached_coord):
            self._cached_value = None

        if self._cached_weights is not None and \
                not np.all(np.equal(self.cascmap.weights, self._cached_weights)):
            self._cached_weights = None

        if self._cached_value is None or self._cached_weights is None:
            self._cached_coordinates = energy
            self._cached_value = self._interpolate(energy * (1. + bias))
            self._cached_weights = self.cascmap.weights

        if self.add_primary:
            result += self._cached_value.to(result.unit)
        else:
            result = self._cached_value.to(result.unit)

        return result

    def copy(self):
        """A shallow copy"""
        new = copy.copy(self)
        return new

    @property
    def parameters(self):
        return (
                Parameters([self.rotation, self.tmax, self.bias])
                + self._intrinsic_spectral_model.parameters
                + self._ebl.parameters
        )

    def to_dict(self):
        data = super().to_dict()
        data["name"] = self.name
        data["type"] = data.pop("type")
        data["spectral_model"] = self.spectral_model.to_dict()
        data["parameters"] = Parameters([self.rotation, self.tmax]).to_dict()

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        return data

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    # TODO covariance handling


class PriorSpectrumDatasetOnOff(SpectrumDatasetOnOff):
    """Spectrum On Off dataset with possibility to add additional term / prior to the likelihood"""

    def __init__(self, *args, **kwargs):
        self._llh_fermi_interp = kwargs.pop("llh_fermi_interp", None)
        self._ref_energy = kwargs.pop("ref_energy", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_spectrum_dataset_fermi_interp(
        cls, dataset, llh_fermi_interp=None, ref_energy=None, ref_index=None
    ):
        """Create spectrum dataseton off from another dataset.

        Parameters
        ----------
        dataset : `SpectrumOnOffDataset`
            Spectrum dataset defining counts, edisp, exposure etc.
        acceptance : `~numpy.array` or float
            Relative background efficiency in the on region.
        acceptance_off : `~numpy.array` or float
            Relative background efficiency in the off region.
        counts_off : `~gammapy.maps.RegionNDMap`
            Off counts spectrum . If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """
        return cls(
            models=dataset.models,
            counts=dataset.counts,
            exposure=dataset.exposure,
            counts_off=dataset.counts_off,
            edisp=dataset.edisp,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=dataset.acceptance,
            acceptance_off=dataset.acceptance_off,
            gti=dataset.gti,
            name=dataset.name,
            meta_table=dataset.meta_table,
            llh_fermi_interp=llh_fermi_interp,
            ref_energy=ref_energy
        )

    @property
    def llh_fermi_interp(self):
        return self._llh_fermi_interp

    @property
    def llh_fermi(self):
        return self._llh_fermi

    @llh_fermi_interp.setter
    def llh_fermi_interp(self, llh_fermi_interp):
        self._llh_fermi_interp = llh_fermi_interp

    @property
    def ref_energy(self):
        return self._ref_energy

    @ref_energy.setter
    def ref_energy(self, ref_energy):
        self._ref_energy = ref_energy

    def get_llh_fermi(self, amplitude=None, index=None, lambda_=None, reference=None, set_attribute=True):

        if amplitude is None:
            amplitude = self.models['casc'].parameters['amplitude'].quantity
        if index is None:
            index = self.models['casc'].parameters['index'].quantity
        if lambda_ is None:
            lambda_ = self.models['casc'].parameters['lambda_'].quantity
        if reference is None:
            reference = self.models['casc'].parameters['reference'].quantity

        amplitude = amplitude.to('MeV-1 cm-2 s-1').value
        reference = reference.to('MeV').value
        index = index.value
        cutoff = 1. / lambda_.to("TeV-1").value

        if not isinstance(amplitude, Iterable):
            amplitude = np.array([amplitude])

        if not isinstance(index, Iterable):
            index = np.array([index])

        if not isinstance(cutoff, Iterable):
            cutoff = np.array([cutoff])

        # correct amplitude for different reference energy
        # for which the fermi likelihood scan was performed
        if self._ref_energy is not None:
            if amplitude.size == index.size:
                amplitude *= (self._ref_energy / reference) ** -index
            elif index.size == 1:
                amplitude *= (self._ref_energy / reference) ** -index[0]
            else:
                raise ValueError("Incompatible shapes between index and amplitude")

        # build coordinate array for interpolation of the shape cutoff, index, log10(amplitude)
        # should have shape (n_coords, 3)
        coord_list = [cutoff, index, np.log10(amplitude)]
        if cutoff.size == index.size == amplitude.size == 1:
            result = -2. * self._llh_fermi_interp([cutoff[0], index[0], np.log10(amplitude[0])])[0]

        else:
            sizes = [t.size for t in coord_list]
            max_size = np.max(sizes)
            coords = np.array([[coord_list[i][j if j < sizes[i] - 1 else 0] for i in range(len(coord_list))] for j in range(max_size)])
            result = -2. * self._llh_fermi_interp(coords)

        if set_attribute:
            self._llh_fermi = result

        return result 

    def stat_sum(self):
        # calculate -2 * log likelihood
        # from dataset
        log_like = super().stat_sum()

        # add the fermi likelihood from interpolation
        if self._llh_fermi_interp is not None:
            #amplitude = self.models['casc'].parameters['amplitude'].quantity.to('MeV-1 cm-2 s-1').value
            #reference = self.models['casc'].parameters['reference'].quantity.to('MeV').value
            #index = self.models['casc'].parameters['index'].value

            # correct amplitude for different reference energy
            # for which the fermi likelihood scan was performed
            #if self._ref_energy is not None:
                #amplitude *= (self._ref_energy / reference) ** -index

            #cutoff = 1. / self.models['casc'].parameters['lambda_'].quantity.to("TeV-1").value

            ##self._llh_fermi = -2. * self._llh_fermi_interp([cutoff, index, np.log10(amplitude)])[0]

            self.get_llh_fermi()

            log_like += self._llh_fermi

        # total likelihood
        return log_like


class PriorDataset(Datasets):
    """Spectrum On Off dataset with possibility to add additional term / prior to the likelihood"""

    def __init__(self, datasets=None, llh_fermi_interp=None, **kwargs):
        super().__init__(datasets, **kwargs)
        self._llh_fermi_interp = llh_fermi_interp
        self._llh_fermi = None

    @property
    def llh_fermi_interp(self):
        return self._llh_fermi_interp

    @property
    def llh_fermi(self):
        return self._llh_fermi

    @llh_fermi_interp.setter
    def llh_fermi_interp(self, llh_fermi_interp):
        self._llh_fermi_interp = llh_fermi_interp

    def stat_sum(self):
        # calculate -2 * log likelihood
        # from dataset
        log_like = super().stat_sum()

        # add the fermi likelihood from interpolation
        if self._llh_fermi_interp is not None:
            amplitude = self.models['casc'].parameters['amplitude'].quantity.to('MeV-1 cm-2 s-1').value
            index = self.models['casc'].parameters['index'].value
            cutoff = 1. / self.models['casc'].parameters['lambda_'].quantity.to("TeV-1").value

            self._llh_fermi = -2. * self._llh_fermi_interp([cutoff, index, np.log10(amplitude)])[0]

            log_like += self._llh_fermi

        # total likelihood
        return log_like

# gammapy 0.17
#class SkyDiffuseCascadeCube(SkyModelBase):
class SkyDiffuseCascadeCube(SpatialModel):
    """Cube sky map model for electromagnetic cascades.

    Parameters
    ----------
    cascmap : `~simCRpropa.cascmaps.CascMap`
        Cascade Map template

    spectral_model : `SpectralModel`
        Spectral model of the injected particles.

    rotation: float
        Rotation angle of the cascade template

    tmax: float
        Maximum delay time in years allowed for the cascade.

    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.
    """

    tag = "SkyDiffuseCascadeCube"
    rotation = Parameter("rotation", 0., unit="deg", frozen=True)
    tmax = Parameter("tmax", 1e7, unit="yr", frozen=True)

    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
            self,
            cascmap,
            spectral_model,
            rotation=rotation.quantity,
            tmax=tmax.quantity,
            interp_kwargs=None,
            apply_irf=None,
            name=None,
    ):

        self.cascmap = cascmap
        self._spectral_model = spectral_model
        self._energy_unit_spectral_model = energy_unit_spectral_model
        self._name = name

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

        self._cached_value = None
        self._cached_weights = None
        self._cached_coordinates = (None, None, None)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        super().__init__(tmax=tmax, rotation=rotation)

    @property
    def name(self):
        return self._name

    def _interpolate(self, lon, lat, energy):
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
            "energy": energy,
        }
        return self.cascmap.add_primary_to_casc().interp_by_coord(coord, **self._interp_kwargs)

    def evaluate(self, lon, lat, energy, **kwargs):
        """Evaluate model at given coordinates"""

        rotation = kwargs.pop("rotation")
        tmax = kwargs.pop("tmax")

        # change max delay time
        if not tmax == self.cascmap.tmax:
            self.cascmap.tmax = tmax

        # change rotation angle
        # and apply rotation
        if not rotation == self.cascmap.angle:
            self.cascmap.angle = rotation

        # change spectral weights
        self.cascmap.apply_spectral_weights(self._inj_spec, smooth=True, **kwargs)

        is_cached_coord = [
            _ is coord for _, coord in zip((lon, lat, energy), self._cached_coordinates)
        ]

        # reset cache
        if not np.all(is_cached_coord):
            self._cached_value = None

        if self._cached_weights is not None and \
                not np.all(np.equal(self.cascmap.weights, self._cached_weights)):
            self._cached_weights = None

        if self._cached_value is None or self._cached_weights is None:
            self._cached_coordinates = (lon, lat, energy)
            self._cached_value = self._interpolate(lon, lat, energy)
            self._cached_weights = self.cascmap.weights

        return u.Quantity(self._cached_value, self.cascmap.casc_obs.unit, copy=False)

    def copy(self):
        """A shallow copy"""
        new = copy.copy(self)
        return new

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.cascmap.casc_obs.geom.center_skydir

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return np.max(self.cascmap.casc_obs.geom.width) / 2.0

    @property
    def parameters(self):
        return (
                Parameters([self.rotation, self.tmax])
                + self.spectral_model.parameters
        )

    def to_dict(self):
        data = super().to_dict()
        data["name"] = self.name
        data["type"] = data.pop("type")
        data["spectral_model"] = self.spectral_model.to_dict()
        data["parameters"] = Parameters([self.rotation, self.tmax]).to_dict()

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        return data

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    # TODO covariance handling


