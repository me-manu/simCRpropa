import copy
from collections import OrderedDict
import yaml
import h5py
import logging
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM, Planck15
from simCRpropa import rotations as rot
from gammapy.maps import Map, MapAxis
from scipy.integrate import simps
from scipy.ndimage import rotate
from scipy.interpolate import UnivariateSpline, interp1d
from gammapy.modeling.models.cube import SkyModelBase
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.scripts import make_name


def stack_results_lso(infile, outfile, **kwargs):
    """
    Stack the results of bin-by-bin simulations
    for simulations with a large sphere observer
    and combine them into arrays for injected energy,
    observed energy, time delay, angular separation, sky coordinates,
    and particle ID

    Parameters
    ----------
    infile: str
        path to input hdf5 file
    outfile: str
        path to output hdf5 file

    {options}

    theta_obs: float
        angle in deg between observer and jet axis (default: 0. deg)
    dgrp: str
        name of hdf5 group where data is stored (default: 'simEM')
    entries: list of str
        list of parameters that will be combined (default: [E0,E,D,X,Px,P0x,ID,ID1])
    entries_stack: list of str
        entries that will be stacked (instead of concatenated, default: [X,Px,P0x])
    entries_save: list of str
        entries that will saved (default: [E0,E,dt,dtheta,Protsph,ID,ID1])
    use_cosmo: bool
        if True, use crpropa or astropy cosmology module to calculate source distance from redshift
        Note: if crpropa is not installed, astropy module might deliver slightly different results
        and thus the calculation of the time delay might be affected
    DSource: float or None
        if use_cosmo is False, source distance will be taken from config dict if available,
        otherwise DSource (in Mpc) will be used

    Returns
    -------
    List of dictionaries with `~numpy.ndarray`s that contain the data and configuration
    """

    kwargs.setdefault('theta_obs', 0.)
    kwargs.setdefault('dgrp', 'simEM')
    kwargs.setdefault('entries', ['E0','E','D','X','Px','P0x','ID','ID1'])
    kwargs.setdefault('entries_stack', ['X','Px','P0x'])
    kwargs.setdefault('entries_save', ['E0','E','dt','Protsph','ID','ID1'])
    kwargs.setdefault('use_cosmo', True)
    kwargs.setdefault('Dsource', None)

    combined = h5py.File(infile, 'r+')
    config = yaml.safe_load(combined[kwargs['dgrp']].attrs['config'])
    kwargs['theta_jet'] = config['Source']['th_jet']

    # init hdf5 file
    h = h5py.File(outfile, 'a')
    if kwargs['dgrp'] in h.keys(): # overwrite group if already exists
        del h[kwargs['dgrp']]
    grp = h.create_group(kwargs['dgrp'])
    grp.attrs["config"] = yaml.safe_dump(config, default_flow_style=False)

    data = {}
    # combine the data from all energy bins
    logging.info("Combining data from all energy bins ...")

    if isinstance(config['Source']['Emin'], float):
        n_ebins = config['Source']['Esteps']-1
    else:
        n_ebins = len(config['Source']['Emin'])

    for ie in range(n_ebins):
        eb = 'Ebin{0:03n}'.format(ie)
        for k in kwargs['entries']:
            ki = 'simEM/{1:s}/{0:s}'.format(eb,k)

            if not ie:
                data[k] = combined[ki][()]
            else:
                if k in kwargs['entries_stack']:
                    data[k] = np.hstack([data[k],
                                         combined[ki][()]])
                else:
                    data[k] = np.concatenate([data[k],
                                              combined[ki][()]])

    for k in ['intspec/Ecen', 'intspec/weights']:
        logging.info("Saving {0} to {1:s}...".format(k, outfile))
        grp.create_dataset(k, data=combined['simEM/' + k],
                           dtype=combined['simEM/' + k].dtype,
                           compression="gzip")
    combined.close()
    logging.info("Done.")

    # rotate positional vectors
    logging.info("Calculating vector rotations and "\
                 "applying cuts for jet axis and observer ...")

    # unit vector to observer
    try:
        xx0norm = (data['X'] - data['X0']) / np.linalg.norm(data['X'] - data['X0'], axis=0)
    except KeyError:
        xx0norm = (data['X']) / np.linalg.norm(data['X'], axis=0)
    # project momentum vector into observer's coordinate system
    pnew = rot.project2observer(data['Px'], xx0norm, axis=0)
    # get pnew in spherical coordinates
    pnewsph = rot.car2sph(-pnew)
    # project initial momentum vector into observer's coordinate system
    p0new = rot.project2observer(data['P0x'], xx0norm, axis=0)
    # Calculate the mask for initial momentum
    # vectors given jet observation and opening angle
    mask = rot.projectjetaxis(p0new,
                              jet_opening_angle=kwargs['theta_jet'],
                              jet_theta_angle=kwargs['theta_obs'],
                              jet_phi_angle=0.)

    data['Protsph'] = np.vstack([pnewsph[0,:],
                                 np.rad2deg(pnewsph[2,:] * np.sin(pnewsph[1,:])),
                                 np.rad2deg(pnewsph[2,:] * np.cos(pnewsph[1,:])) + 90.
                                 ])

    logging.info("Done.")

    # compute time delay in years
    logging.info("Calculating time delay and angular separation...")
    # from either cosmology ...
    if kwargs['use_cosmo']:
        try:
            Dsource = crpropa.redshift2ComovingDistance(config['Source']['z']) * u.m.to('Mpc')
        except:
            # standards in CRPropa, see
            # https://github.com/CRPropa/CRPropa3/blob/master/include/crpropa/Cosmology.h
            cosmo = FlatLambdaCDM(H0=67.3, Om0=0.315)
            Dsource = cosmo.comoving_distance(config['Source']['z']).value # in Mpc
    else:
        # crpropa distance was saved but in m, convert to Mpc
        if 'ComovingDistance' in config['Source']:
            Dsource = config['Source']['ComovingDistance'] * u.m.to('Mpc')
        else:
            Dsource = kwargs['Dsource']
    logging.info("Using Dsource = {0:.5e} Mpc".format(Dsource))

    data['dt'] = (data['D'] - Dsource)
    data['dt'] *= (u.Mpc.to('m') * u.m / c.c).to('yr').value

    # save to an hdf5 file
    logging.info("Saving {0} to {1:s}...".format(kwargs['entries_save'], outfile))
    for k in kwargs['entries_save']:
        if k in grp.keys():  # overwrite group if already exists
            del grp[k]
        if 'ID' in k:
            dtype = 'i8'
        else:
            dtype = 'f8'

        if k == 'Protsph':
            grp.create_dataset(k, dtype = dtype,
                               data = data[k][:,mask], compression="gzip")
        else:
            grp.create_dataset(k, dtype = dtype,
                               data = data[k][mask], compression="gzip")
    h.close()
    logging.info("Done.")

    data['mask'] = mask
    return data, config


class HistPrimary(object):
    """
    Helper class for a histogram of true and observed energy
    for the primary gamma-ray emission of the EM cascade
    """
    def __init__(self, hist_prim, edges_obs, edges_true):
        """

        :param hist_prim:
        :param edges_obs:
        :param edges_true:
        """
        self._data_orig = hist_prim * u.dimensionless_unscaled
        self._energy = MapAxis(edges_obs, interp='log', name='energy', node_type='edges')
        self._energy_true = MapAxis(edges_true, interp='log', name='energy_true', node_type='edges')
        self._data = copy.deepcopy(self._data_orig)

    @property
    def data(self):
        return self._data
    @property
    def energy(self):
        return self._energy
    @property
    def energy_true(self):
        return self._energy_true

    def copy_data(self):
        return copy.deepcopy(self._data_orig)

    def get_obs_spectrum(self, energy=None, **kwargs):
        """
        Get the observed primary spectrum as a function of observed energy
        in units of weights / energy

        Parameters
        ----------
        energy: None or `~astropy.Quantity`
            array with energies as where spectrum is evaluated
            using a 1D spline interpolation

        :return: array
            Array with fluxes per energy as `~astropy.Quantity`
        """
        kwargs.setdefault('s',0.)
        kwargs.setdefault('k',1)
        kwargs.setdefault('ext','extrapolate')

        # sum over true energy axis
        dn_de = self._data.sum(axis=0)

        # divide by observed bin width
        dn_de /= self._energy.bin_width

        # if energy array is given,
        # interpolate
        if energy is not None:
            dn_de.value[dn_de.value == 0.] = np.full(np.sum(dn_de.value == 0.), 1e-40)
            interp = UnivariateSpline(np.log(self._energy.center.value),
                                      np.log(dn_de.value),
                                      **kwargs)
            dn_de_interp = np.exp(interp(np.log(energy.to(self._energy.unit).value)))
            dn_de_interp *= dn_de.unit
        else:
            dn_de_interp = dn_de

        return dn_de_interp

    # TODO: read, write methods


class CascMap(object):
    """
    Class for creation and manipulation
    of n-dim gammapy wcs map with crpropa output for a
    cascade and potentially histogram for intrinsic spectrum.
    """

    def __init__(self,
                 hist_casc,
                 edges,
                 skycoord,
                 hist_prim=None,
                 edges_prim=None,
                 steps=10,
                 binsz=0.02,
                 width=6.,
                 redshift=None
                 ):
        """

        :param hist_casc:
        :param edges:
        :param skycoord:
        :param hist_prim:
        :param edges_prim:
        :param steps:
        :param binsz:
        :param width:
        """
        # get axes not related to sky coordinates
        axes_edges = copy.deepcopy(edges)
        axes_edges.pop('lon')
        axes_edges.pop('lat')
        axes = OrderedDict()
        for k, v in axes_edges.items():
            axes[k] = MapAxis(v, name=k, node_type='edges', interp='lin' if k=='t_delay' else 'log')

        # create the map
        # axes order reversed so that it can be filled with the
        # histogram

        logging.info("Building and filling the map ...")
        self._m = Map.create(binsz=binsz,
                             width=width,
                             skydir=skycoord,
                             axes=list(axes.values())[::-1])
        self._m.data = hist_casc

        if hist_prim is not None:
            self._primary = HistPrimary(hist_prim=hist_prim,
                                        edges_true=edges_prim[0],
                                        edges_obs=edges_prim[1]
                                        )
        else:
            self._primary = None


        # 2d array for integration of injected energy
        self._einj = []
        for i, emin in enumerate(edges['energy_true'][:-1].value):
            self._einj.append(np.logspace(np.log10(emin),
                                          np.log10(edges['energy_true'][i+1].value),
                                          steps))

        self._einj = np.array(self._einj) * edges['energy_true'].unit
        self._weights = np.ones_like(self._m.geom.get_axis_by_name('energy_true').center.value) * \
                        u.dimensionless_unscaled

        self._tmax = edges['t_delay'].max() * u.yr
        self._casc = self._m.sum_over_axes(['t_delay'])

        self._casc_map = self._m.copy()

        # the observed cascade flux, after weights are applied, this will have units of
        # weights.units / eV / sr * eV
        self._casc_obs = self._casc.sum_over_axes(['energy_true'])
        self._casc_obs_bin_volume = self._casc_obs.geom.bin_volume()

        # spatial bin volume
        # which can be applied to casc_obs
        self._spatial_bin_size = self._casc_obs.sum_over_axes(['energy']).geom.bin_volume()

        # energy axes
        self._energy = self._casc.geom.get_axis_by_name('energy')
        self._energy_true = self._casc.geom.get_axis_by_name('energy_true')

        # rotation angle
        self._angle = 0. * u.deg

        # redshift
        self._z = redshift
        logging.info("Done.")

    def apply_time_weights(self, look_back_times=None, weights=None, interpolation_type='nearest'):
        """
        Apply weights to the time bins to emulate a source light curve

        Parameters
        ----------
        :param look_back_times:  array-like or None, optional
            An array of look back times, i.e., 0 corresponds to today and values > 0 correspond to the past.
            Should be in the same units as the t_delay axis, i.e., in years
        :param weights: array-like or None, optional
            An array of weights corresponding to the look back times. These values should be
            given with respect to some average flux value which can be set by the apply_spectral_weights method,
            but in any case, they are normalized to their mean.
        :param interpolation_type: str
            Either 'nearest' or 'linear'. Use 'nearest' if weights are extracted, e.g., from a gamma-ray light curve
            where the flux in a time bin is the average counts collected in that bin. Use 'linear' if the weights
            are truly of differential type, e.g., when extracted from a radio light curve.

        Notes
        -----
        * If weights are None then they will be set such that they are constant up to the maximum
        delay time.
        * the weights will be normalized to their mean, so that they indicate deviation from mean flux
        * weights outside the range defined by look_back_times will be set to zero
        """
        logging.debug("Applying time weights ...")
        # get the time delay axis
        t_axis = self._m.geom.get_axis_by_name('t_delay')

        if look_back_times is None or weights is None:
            look_back_times = [0., self._tmax.to(t_axis.unit).value]
            weights = [1., 1.]

        interp = interp1d(look_back_times, weights / np.mean(weights),
                                  fill_value=0.,
                                  bounds_error=False,
                                  kind=interpolation_type
                                  )
        # interp weights over time delay axis
        # TODO this should probably be replaced by oversampling, if interpolation is not nearest

        # interpolate weights over the time axis of the cascade
        weights_interp = interp(t_axis.center)

        # Now we want to calculate the average cascade flux received within some time Window T_obs
        # which we calculate through the integral
        # T_{obs}^{-1} \int_{-\infty}^0 weights (t) dt \int_{t - T_{obs}}^{t} K(\tau) d\tau
        # where K is the cascade flux at delay time \tau
        # We approximate the second integral by assuming that K is constant over T_obs,
        # hence
        # \int_{t - T_{obs}}^{t} K(\tau) d\tau \approx T_{obs} K(t)
        # simplifying and discretizing the remaining integral, one finds for the average flux
        # \sum_i weights(t_i) K(t_i) \Delta t_i
        # this calculation is performed by the next two lines:
        self._casc_map = self._m * \
                         weights_interp[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        self._casc = self._casc_map.sum_over_axes(['t_delay'])
        # self._casc now contains the time averaged flux
        # given some source history until now (t=0)
        logging.debug("... Done.")

    def sum_until_tmax(self):
        """
        Sum the cascade map over the delay axis up
        to some time tmax and update self._casc with result

        :param tmax:
        :return:
        """
        t = self._m.geom.get_axis_by_name('t_delay')
        mask = t.edges[1:] <= self._tmax.to(t.unit)
        idx = np.argmax(t.edges[1:][mask])
        # sum over axis
        self._casc = self._m.slice_by_idx({'t_delay': slice(0, idx + 1)}).sum_over_axes(['t_delay'])

        # update observed cascade flux
        self._casc_obs = self._casc.sum_over_axes(['energy_true'])
        self._casc_obs_bin_volume = self._casc_obs.geom.bin_volume()

    @property
    def m(self):
        return self._m

    @property
    def casc(self):
        return self._casc

    @property
    def casc_obs(self):
        return self._casc_obs

    @property
    def casc_obs_bin_volume(self):
        return self._casc_obs_bin_volume

    @property
    def tmax(self):
        return self._tmax

    @tmax.setter
    def tmax(self, tmax):
        self._tmax = tmax.to('yr')
        #self.sum_until_tmax()
        self.apply_time_weights()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle.to('deg')
        self.rotation(angle=self._angle)

    @property
    def weights(self):
        return self._weights

    @property
    def primary(self):
        return self._primary

    @property
    def z(self):
        return self._z

    @property
    def energy(self):
        return self._energy

    @property
    def energy_true(self):
        return self._energy_true

    @staticmethod
    def gen_from_hd5f(infile, skycoord,
                      dgrp='simEM',
                      width=6.,
                      ebins=41,
                      binsz=0.02,
                      id_detection=22,
                      lightcurve=None,
                      ):
        # TODO allow to supply map WCS geometry?
        # TODO include some print / logging statements
        """
        generate gammapy.map from hd5f file
        bin boundaries are set automatically
        to hardcoded reasonable values

        parameters
        ----------
        infile: str or list of str
            path(s) to hd5f file(s) created with the stack_results_lso function.
            If list of files, true energy bins must not overlap but smoothly connect.

        skycoord: `~astropy.coordinates.skycoord` object
            sky coordinates for observation


        kwargs
        ------
        dgrp: str
            Name of data group of h5df file
        width: float
            width of roi, in degrees, used for binning in phi and theta
            (default: 6.)
        binsz: float
            with of each pixel in degrees for resulting histogram
            which will be used for phi and theta binning
            (default: 0.02, motivated from minimum cta psf)
        ebins: int or array-like
            total number of bins of observed energy
        lightcurve: None or float
            if not None, then this is the maximum time in years of the light curve
            which will be binned according to resolution of simulation.
            Beware: this might result in a large number of time bins!
        id_detection: int
            identifier for detected particles, following CERN MC scheme (22 = photons)
        """
        # TODO allow for list of files covering different energy ranges
        # TODO simply concatenate data arrays and update config dict?

        if isinstance(infile, str):
            infile = [infile]

        # edges for histogram
        edges = OrderedDict({})


        # read in data values
        values = {}
        tmin, tmin_data, tmax_data = [], [], []

        for i, filename in enumerate(infile):
            logging.info(f"Reading file {filename:s}")
            hfile = h5py.File(filename, 'r+')
            data = hfile[dgrp]
            config = yaml.safe_load(data.attrs['config'])

            # get the minimum resolution in weeks
            min_resol = (config['Simulation']['minStepLength'] * u.pc / c.c.to('pc / s')).to('yr')
            tmin.append(min_resol.value)
            tmax_data.append(data['dt'][()].max())
            tmin_data.append(data['dt'][()].min())
            logging.info(f"time resolution of the simulation was {min_resol:.3f} = {min_resol.to('day'):.3f} ")
            if tmin_data[-1] < -1. * tmin[-1]:
                logging.error("Minimum time delay is smaller then -1 * time resolution, check assumed source distance")

            logging.info("Injected particle ID: {0}".format(config['Source']['Composition']))
            # injected energy bins
            # are determined already here
            if isinstance(config['Source']['Emin'], float):
                energy_true = np.logspace(np.log10(config['Source']['Emin']),
                                          np.log10(config['Source']['Emax']),
                                          config['Source']['Esteps'])
            else:
                energy_true = np.append(config['Source']['Emin'],
                                        config['Source']['Emax'][-1])
            if not i:
                edges['energy_true'] = energy_true
                n_injected_particles = data['intspec/weights'][()]
            else:
                # check if true energy edges overlap, not allowed
                # otherwise append / prepend
                if energy_true[0] == edges['energy_true'][-1]:
                    edges['energy_true'] = np.append(edges['energy_true'], energy_true[1:])
                    n_injected_particles = np.append(n_injected_particles, data['intspec/weights'][()])
                elif energy_true[-1] == edges['energy_true'][0]:
                    edges['energy_true'] = np.append(energy_true[:-1], edges['energy_true'])
                    n_injected_particles = np.append(data['intspec/weights'][()], n_injected_particles)
                else:
                    raise ValueError("True energy axis of different files must connect to each other!")

            if not i:
                values['energy_true'] = data['E0'][()]
                values['energy'] = data['E'][()]
                values['t_delay'] = data['dt'][()]
                values['lon'] = data['Protsph'][1,:]
                values['lat'] = data['Protsph'][2,:]
                values['id_obs'] = data['ID'][()]
                values['id_parent'] = data['ID1'][()]

            else:
                values['energy_true'] = np.append(values['energy_true'], data['E0'][()])
                values['energy'] = np.append(values['energy'], data['E'][()])
                values['t_delay'] = np.append(values['t_delay'], data['dt'][()])
                values['lon'] = np.append(values['lon'], data['Protsph'][1,:])
                values['lat'] = np.append(values['lat'], data['Protsph'][2,:])
                values['id_obs'] = np.append(values['id_obs'], data['ID'][()])
                values['id_parent'] = np.append(values['id_parent'], data['ID1'][()])

            hfile.close()

        edges['energy_true'] *= u.eV

        if not np.all(np.array(tmin) == tmin[0]):
            logging.warning("Not all time resolutions are equal, selecting maximum")

        # time resolution
        tmin = np.max(tmin)

        # minimum time within time resolution, set all values < 0. to zero
        if -1. * tmin < np.min(tmin_data) <= 0.:
            logging.info("minimum delay time <= 0. but within resolution, forcing to zero")
            values['t_delay'][values['t_delay'] <= 0.] = np.zeros(np.sum(values['t_delay'] <= 0))
            insert_tmin = 0.
        else:
            insert_tmin = np.min(tmin_data)

        if isinstance(lightcurve, float):
            edges['t_delay'] = np.insert(np.arange(2. * tmin, lightcurve, 2. * tmin),
                                         0,
                                         insert_tmin)
            # and insert all remaining time bins
            edges['t_delay'] = np.append(edges['t_delay'],
                                         np.logspace(np.log10(np.max([edges['t_delay'].max(), 10.])), 8., 8)
                                         )

        else:
            first_bin = np.max([tmin, 1.])
            edges['t_delay'] = np.append(
                np.insert(np.logspace(np.log10(first_bin),7,8),
                          0,
                          insert_tmin),
                np.max(tmax_data)
            )

        edges['t_delay'] *= u.yr
        logging.info("Time bins: {0}".format(edges['t_delay']))


        # observed energy bins
        if isinstance(ebins, int):
            edges['energy'] = np.logspace(np.log10(values['energy'].min()),
                                          np.log10(values['energy'].max()),
                                          ebins) * u.eV

        elif isinstance(ebins, list) or isinstance(ebins, tuple):
            edges['energy'] = np.array(ebins)

        elif isinstance(ebins, np.ndarray):
            edges['energy'] = ebins

        # phi
        binsz *= u.deg / u.pixel
        width *= u.deg
        nbins = np.ceil(2.*width/2./binsz).astype(np.int)
        edges['lon'] = np.linspace(-width.value/2.,
                                   width.value/2.,
                                   nbins.value + 1) * width.unit
        # theta
        edges['lat'] = np.linspace(90.-width.value/2.,
                                   90. + width.value/2.,
                                   nbins.value + 1) * width.unit

        # create a numpy histogram for the cascade
        # and potentially for the source

        # first make sure ordering is correct
        # needs to be in order t_delay, energy_true, energy, lon, lat
        keys = ['t_delay', 'energy_true', 'energy', 'lon', 'lat']
        edges = OrderedDict((k, edges[k]) for k in keys)

        hist_casc, hist_prim, edges_prim = CascMap.build_nd_histogram(edges,
                                                                      values,
                                                                      id_detection=id_detection,
                                                                      id_injected=config['Source']['Composition'],
                                                                      config=config)

        # divide histogram by injected number of particles
        hist_casc /= n_injected_particles[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        if hist_prim is not None:
            hist_prim /= n_injected_particles[:, np.newaxis]

        return CascMap(hist_casc,
                       edges,
                       skycoord,
                       hist_prim=hist_prim,
                       edges_prim=edges_prim,
                       binsz=binsz,
                       width=width,
                       redshift=config['Source']['z']
                       )

    @staticmethod
    def build_nd_histogram(edges, values, config=None, id_detection=22, id_injected=None):
        """
        Build an n x d dimensional histogram from cascade simulations

        :param edges: dict
            dict with bin edges
        :param values: dict
            dict with raw counts
        :param id_detection: int
            id of detected particle
        :param id_injected: int or None
            id of injected particle
        :param config: dict or None
            dict with simulation configuration
        :return:
        """
        # cascade condition
        mc = (values['id_obs'] == id_detection) & \
             (values['id_parent'] != id_injected)
        # build data cube for cascade
        if np.sum(mc):
            data_casc = np.array([values[k][mc] for k in edges.keys()])
            # build the histogram
            logging.info("Building the cascade histogram ...")
            logging.info("Bin shapes: {0}".format([v.shape for v in edges.values()]))
            hist_casc, _ = np.histogramdd(data_casc.T, bins=list(edges.values()))
        else:
            raise ValueError("no events pass cascade criterion")

        # build data cube for primary spectrum
        # injected spectrum condition
        hist_prim = None
        edges_prim = None
        if id_injected is not None:
            logging.info("Building the primary histogram ...")
            mi = (values['id_obs'] == id_detection) & (values['id_parent'] == id_injected)
            if np.sum(mi):
                keys = ['energy_true', 'energy']
                data_primary = np.array([values[k][mi] for k in keys])
                if isinstance(config, dict):
                    # TODO: accounted for redshift, but is it correct?
                    # edges['e_true'] contain the injected energies
                    # edges['e_true'] / ( 1 + z) are the observed energies
                    hist_prim, edges_prim = np.histogramdd(data_primary.T,
                                                           bins=(edges['energy_true'],
                                                                 edges['energy_true'] / (1. + config['Source']['z']))
                                                           )
                else:
                    hist_prim, edges_prim = np.histogramdd(data_primary.T,
                                                           bins=(edges['energy_true'], edges['energy']))
                edges_prim = [e * edges['energy_true'].unit for e in edges_prim]
            else:
                logging.warning("no events pass primary spectrum criterion")
        return hist_casc, hist_prim, edges_prim

    def reset_casc(self):
        self._tmax = self._m.geom.get_axis_by_name('t_delay').edges.max()
        self._casc = self._m.sum_over_axes(['t_delay'])
        self._casc_obs = self._casc.sum_over_axes(['energy_true'])

    def lumi_iso(self, injspec, doppler=1., cosmo=Planck15):
        """
        Compute the equivalent isotropic luminosity for the
        assumed injection spectrum.

        :param injspec: function pointer
            function that takes energy as Quantity and returns flux per energy
        :param doppler: float
            assumed Doppler factor
        :param cosmo: `~astropy.cosmology.FlatLambdaCDM`
            assumed cosmology
        :return:
        """
        cen = self._casc.geom.get_axis_by_name('energy_true').center
        f = injspec(cen)
        lumi_iso = simps(f.value * cen.value, cen.value) * f.unit * cen.unit**2.
        if self._z is not None:
            lumi_iso *= cosmo.luminosity_distance(self._z).to('cm')**2. * 4. * np.pi
        else:
            return ValueError("Could not compute luminosity since redshift not provided.")
        lumi_iso *= doppler ** -4.
        return lumi_iso.to('erg s-1')

    def _compute_spectral_weights(self, injspec):
        """
        Set weights to compute cascade for an arbitrary spectrum.
        Spectrum should take energies in eV and return flux units in terms of eV.

        :param injspec: function pointer
            function that takes energy as Quantity and returns flux per energy
        :return:
        """

        # flux of new injected spectrum integrated in
        # bins of injected spectrum
        # as update for weights
        f = injspec(self._einj)

        target_unit = self._energy_true.unit.to_string()

        # make sure that the right energy unit is used
        funit_split = f.unit.to_string().split('/')
        if not funit_split[0] == "1.":
            funit_split[0] = target_unit + " / "
            f = f.to(' '.join(funit_split))

        # make sure that the right energy unit is used
        if not ' ' + target_unit in f.unit.to_string():
            units = f.unit.to_string().split()
            for i, t in enumerate(units):
                ui = t.strip('/()')
                if u.Unit(ui).physical_type == 'energy':
                    units[i] = units[i].replace(ui, target_unit)
            target_weight_unit = u.Unit(' '.join(units))
        else:
            target_weight_unit = f.unit

        # compute weights
        weights = simps(f.to(target_weight_unit).value * self._einj.value, np.log(self._einj.value), axis=1)

        # apply units
        weights *= target_weight_unit * self._einj.unit
        return weights


    def apply_spectral_weights(self, injspec):
        """
        Apply weights to compute cascade for an arbitrary spectrum

        :param injspec: function pointer
            function that takes energy as Quantity and returns flux per energy
        :return:
        """
        weights = self._compute_spectral_weights(injspec)
        # weights did not change, return
        if not self._weights.unit == u.dimensionless_unscaled and np.all(np.equal(weights,
                                                                         self._weights)):
            return
        # update
        else:
            self._weights = weights
            if self._primary is not None:
                self._primary._data = self._primary.copy_data() * self._weights

            # the observed flux in units of weights.unit * eV / eV / sr
            # one unit of eV from integration, and the other from the bin volume
            # thus, casc_obs is a differential quantity
            self._casc_obs = \
                (self._casc * self._weights[:, np.newaxis, np.newaxis, np.newaxis]).sum_over_axes(['energy_true'])
            self._casc_obs /= self._casc_obs_bin_volume

    def rotation(self, angle, **kwargs):
        """
        Rotate cascade histogram using scipy.ndimage.rotate

        :param angle: `astropy.Quantity`
            rotation angle in degrees w.r.t. to original map

        :param kwargs: dict
            addtional options passed to scipy.ndimage.rotate
        :return:
        """
        kwargs.setdefault('mode', 'reflect')
        kwargs.setdefault('reshape', 'False')
        kwargs.setdefault('axes', (-2,-1))

        if not self._angle == angle:
            # always rotate w.r.t. original map
            angle_rot = angle.to('deg') - self._angle
            self._casc_obs.data = rotate(self._casc_obs.data,
                                         angle=angle_rot.value,
                                         **kwargs)
            self._angle = angle

    def get_obs_spectrum(self, region=None, add_primary=False):
        """
        Get the spectrum of the current `~CascMap.casc_obs` object
        within a given region

        Parameters
        ----------
        region: `~regions.Region`
             Region (pixel or sky regions accepted).

        Returns
        -------
        spectrum : `~gammapy.maps.RegionNDMap`
            Spectrum in the given region.
        """
        # multiply with spatial bin size (in sr)
        # to obtain the integrated value
        if add_primary:
            #casc_tot =
            spec = (self.add_primary_to_casc() * self._spatial_bin_size).get_spectrum(region=region)
        else:
            spec = (self._casc_obs * self._spatial_bin_size).get_spectrum(region=region)
        return spec

    def add_primary_to_casc(self):
        """
        If the primary spectrum is given, add its flux to the central bin
        of the `~CascMap.casc_obs` object and return the resulting map.

        If no primary spectrum available return `~CascMap.casc_obs`
        directly.

        :return:
        """
        casc_tot = self._casc_obs.copy()

        if self._primary is not None:
            # get the observed flux of the primary spectrum
            dn_de_primary = self._primary.get_obs_spectrum(
                energy=self._casc_obs.geom.get_axis_by_name('energy').center
            )

            # get the central pixel locations
            idx = self._casc_obs.geom.center_pix[:-1]
            idx_int = np.ceil(idx).astype(np.int)

            # add it to the central pixel of cascade map
            # for all observed energy bins
            casc_tot.set_by_pix([idx[0], idx[1]],
                                casc_tot.get_by_pix([idx[0], idx[1]]) * casc_tot.unit + \
                                dn_de_primary / self._spatial_bin_size[idx_int[0], idx_int[1]]
                                )
        return casc_tot

class SkyDiffuseCascadeCube(SkyModelBase):
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
        name=None
    ):

        self.cascmap = cascmap
        self.spectral_model = spectral_model
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
        self.cascmap.apply_spectral_weights(injspec=lambda energy: \
                                            self.spectral_model.evaluate(
                                                energy=energy, **kwargs)
                                            )

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
