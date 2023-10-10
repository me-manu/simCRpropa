import copy
from collections import OrderedDict
import yaml
import h5py
import logging
import numpy as np
import json
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM, Planck15
from simCRpropa import rotations as rot
from gammapy.maps import WcsMap, Map, MapAxis, WcsGeom
from scipy.integrate import simps
from scipy.ndimage import rotate
from scipy.interpolate import UnivariateSpline, interp1d
from regions import CircleSkyRegion
from astropy.coordinates import Angle
from gammapy import __version__ as gpv
from astropy.convolution import Tophat2DKernel, Gaussian2DKernel
from collections import Iterable

if float(gpv.split('.')[1]) > 16 or float(gpv.split('.')[0]) > 0:
    from gammapy.utils.array import scale_cube
else:
    from gammapy.maps import scale_cube


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
    kwargs.setdefault('entries', ['E0', 'E', 'D', 'X', 'Px', 'P0x', 'ID', 'ID1', 'W'])
    kwargs.setdefault('entries_stack', ['X', 'Px', 'P0x'])
    kwargs.setdefault('entries_save', ['E0', 'E', 'dt', 'Protsph', 'ID', 'ID1', 'W'])
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
    def __init__(self, hist_prim, edges_obs_frame, edges_gal_frame):
        """

        :param hist_prim: array-like
            histogram of the primary gamma-ray spectrum per injected particle
        :param edges_observer_frame:
            energy edges in observer's frame
        :param edges_galaxy_frame:
            energy edges in Galactic frame, i.e., observed energies * (1. + z)
        """
        self._data_orig = hist_prim * u.dimensionless_unscaled
        self._energy_obs_frame = MapAxis(edges_obs_frame, interp='log', name='energy_obs_frame', node_type='edges')
        self._energy_gal_frame = MapAxis(edges_gal_frame, interp='log', name='energy_gal_frame', node_type='edges')
        self._data = copy.deepcopy(self._data_orig)

    @property
    def data(self):
        return self._data
    @property
    def energy_obs_frame(self):
        return self._energy_obs_frame
    @property
    def energy_gal_frame(self):
        return self._energy_gal_frame

    def copy_data(self):
        return copy.deepcopy(self._data_orig)

    def get_obs_spectrum(self, energy_obs_frame=None, **kwargs):
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
        dn_de /= self._energy_obs_frame.bin_width

        # if energy array is given,
        # interpolate
        if energy_obs_frame is not None:
            dn_de.value[dn_de.value == 0.] = np.full(np.sum(dn_de.value == 0.), 1e-40)
            interp = UnivariateSpline(np.log(self._energy_obs_frame.center.value),
                                      np.log(dn_de.value),
                                      **kwargs)
            dn_de_interp = np.exp(interp(np.log(energy_obs_frame.to(self._energy_obs_frame.unit).value)))
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
                 redshift=None,
                 smooth_kwargs={'kernel': Tophat2DKernel, 'threshold': 4, 'steps': 50},
                 config=None
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
                                        edges_gal_frame=edges_prim[0],
                                        edges_obs_frame=edges_prim[1]
                                        )
        else:
            self._primary = None

        # 2d array for integration of injected energy
        self._einj = []
        for i, emin in enumerate(edges['energy_injected'][:-1].value):
            self._einj.append(np.logspace(np.log10(emin),
                                          np.log10(edges['energy_injected'][i+1].value),
                                          steps))

        self._einj = np.array(self._einj) * edges['energy_injected'].unit
        e_inj_axis = self._m.geom.axes['energy_injected']
        self._weights = np.ones_like(e_inj_axis.center.value) * \
                        u.dimensionless_unscaled

        self._tmax = edges['t_delay'].max() * u.yr
        self._casc = self._m.sum_over_axes(['t_delay'], keepdims=False)

        self._casc_map = self._m.copy()

        # the observed cascade flux, after weights are applied, this will have units of
        # weights.units / eV / sr * eV
        self._casc_obs = self._casc.sum_over_axes(['energy_injected'], keepdims=False)
        self._casc_obs_bin_volume = self._casc_obs.geom.bin_volume()

        # spatial bin volume
        # which can be applied to casc_obs
        self._spatial_bin_size = self._casc_obs.sum_over_axes(['energy_true'], keepdims=False).geom.bin_volume()

        # energy axes
        self._energy_true = self._casc.geom.axes['energy_true']
        self._energy_injected = self._casc.geom.axes['energy_injected']

        # rotation angle
        self._angle = 0. * u.deg

        # redshift
        self._z = redshift

        # init adaptive smoothing
        self._asmooth = ASmooth(self._casc_obs, **smooth_kwargs)
        logging.info("Done.")

        if config is not None:
            self._config = config

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

        Returns
        --------
        tuple with time delays and corresponding weights

        Notes
        -----
        * If weights are None then they will be set such that they are constant up to the maximum
        delay time.
        * the weights will be normalized to their mean, so that they indicate deviation from mean flux
        * weights outside the range defined by look_back_times will be set to zero
        """
        logging.debug("Applying time weights ...")
        # get the time delay axis
        t_axis = self._m.geom.axes['t_delay']

        if weights is None:
            look_back_times = [0., self._tmax.to(t_axis.unit).value]
            weights = [1., 1.]

        elif look_back_times is not None and not weights.size == t_axis.center.size:
            interp = interp1d(look_back_times, weights / np.mean(weights),
                                      fill_value=0.,
                                      bounds_error=False,
                                      kind=interpolation_type
                                      )
            # interp weights over time delay axis
            # TODO this should probably be replaced by oversampling, if interpolation is not nearest

            # interpolate weights over the time axis of the cascade
            weights = interp(t_axis.center)

        # apply weights directly to different time delays
        else:
            weights = weights

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
                         weights[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        self._casc = self._casc_map.sum_over_axes(['t_delay'], keepdims=False)
        # self._casc now contains the time averaged flux
        # given some source history until now (t=0)
        logging.debug("... Done.")
        return t_axis.center, weights

    def sum_until_tmax(self):
        """
        Sum the cascade map over the delay axis up
        to some time tmax and update self._casc with result

        :param tmax:
        :return:
        """
        t = self._m.geom.axes['t_delay']
        mask = t.edges[1:] <= self._tmax.to(t.unit)
        idx = np.argmax(t.edges[1:][mask])
        # sum over axis
        self._casc = self._m.slice_by_idx({'t_delay': slice(0, idx + 1)}).sum_over_axes(['t_delay'], keepdims=False)

        # update observed cascade flux
        self._casc_obs = self._casc.sum_over_axes(['energy_injected'], keepdims=False)
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

    @property
    def tmin(self):
        return self._tmin

    @tmax.setter
    def tmax(self, tmax):
        self._tmax = tmax.to('yr')
        #self.sum_until_tmax()
        self.apply_time_weights()
        self._weights = np.ones_like(self._m.geom.axes['energy_injected'].center.value) * \
                        u.dimensionless_unscaled

    @tmin.setter
    def tmin(self, tmin):
        self._tmin = tmin.to('yr')
        #self.apply_time_weights()
        #self._weights = np.ones_like(self._m.geom.axes['energy_injected'].center.value) * \
                        #u.dimensionless_unscaled

    #TODO implement tmin correctly!
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
    def energy_true(self):
        return self._energy_true

    @property
    def energy_injected(self):
        return self._energy_injected

    @property
    def config(self):
        return self._config

    @property
    def asmooth(self):
        return self._asmooth

    @staticmethod
    def gen_from_hd5f(infile, skycoord,
                      dgrp='simEM',
                      width=6.,
                      ebins=41,
                      binsz=0.02,
                      id_detection=22,
                      lightcurve=None,
                      smooth_kwargs={'kernel': Tophat2DKernel, 'threshold': 4, 'steps': 50}
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
        lightcurve: None, float, or array-like
            if float, then this is the maximum time in years of the light curve
            which will be binned according to resolution of simulation.
            Beware: this might result in a large number of time bins!
            if array, then these are the light curve bins in years
        id_detection: int
            identifier for detected particles, following CERN MC scheme (22 = photons)
        """
        # TODO allow for list of files covering different energy ranges
        # TODO simply concatenate data arrays and update config dict?

        if isinstance(infile, str):
            infile = [infile]

        binsz, config, edges, n_injected_particles, values, width = CascMap.prep_cascade_hist_from_hdf5(infile,
                                                                                                dgrp=dgrp,
                                                                                                binsz=binsz,
                                                                                                ebins=ebins,
                                                                                                lightcurve=lightcurve,
                                                                                                width=width)

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
                       redshift=config['Source']['z'],
                       config=config,
                       smooth_kwargs=smooth_kwargs
                       )

    @staticmethod
    def prep_cascade_hist_from_hdf5(infile, binsz=0.02, dgrp='simEM', ebins=41, lightcurve=None, width=6.):
        """
        Helper function to create a histogram from a list of hdf5 files
        """

        # edges for histogram
        edges = OrderedDict({})
        # read in data values
        values = {}
        tmin, tmin_data, tmax_data = [], [], []

        for i, filename in enumerate(infile):
            logging.info("Reading file {0:s}".format(filename))
            hfile = h5py.File(filename, 'r+')
            data = hfile[dgrp]
            config = yaml.safe_load(data.attrs['config'])

            # get the minimum resolution in weeks
            min_resol = (config['Simulation']['minStepLength'] * u.pc / c.c.to('pc / s')).to('yr')
            tmin.append(min_resol.value)
            tmax_data.append(data['dt'][()].max())
            tmin_data.append(data['dt'][()].min())

            logging.info("time resolution of the simulation was"
                         "{0:.3f} = {1:.3f}".format(min_resol, min_resol.to('day')))
            if tmin_data[-1] < -1. * tmin[-1]:
                logging.error("Minimum time delay is smaller then -1 * time resolution, check assumed source distance")

            logging.info("Injected particle ID: {0}".format(config['Source']['Composition']))
            # injected energy bins
            # are determined already here
            if isinstance(config['Source']['Emin'], float):
                energy_injected = np.logspace(np.log10(config['Source']['Emin']),
                                          np.log10(config['Source']['Emax']),
                                          config['Source']['Esteps'])
            else:
                energy_injected = np.append(config['Source']['Emin'],
                                        config['Source']['Emax'][-1])
            if not i:
                edges['energy_injected'] = energy_injected
                n_injected_particles = data['intspec/weights'][()]
            else:
                # check if true energy edges overlap, not allowed
                # otherwise append / prepend
                if energy_injected[0] == edges['energy_injected'][-1]:
                    edges['energy_injected'] = np.append(edges['energy_injected'], energy_injected[1:])
                    n_injected_particles = np.append(n_injected_particles, data['intspec/weights'][()])
                elif energy_injected[-1] == edges['energy_injected'][0]:
                    edges['energy_injected'] = np.append(energy_injected[:-1], edges['energy_injected'])
                    n_injected_particles = np.append(data['intspec/weights'][()], n_injected_particles)
                else:
                    raise ValueError("True energy axis of different files must connect to each other!")

            if not i:
                values['energy_injected'] = data['E0'][()]
                values['energy_true'] = data['E'][()]
                values['t_delay'] = data['dt'][()]
                values['lon'] = data['Protsph'][1, :]
                values['lat'] = data['Protsph'][2, :]
                values['id_obs'] = data['ID'][()]
                values['id_parent'] = data['ID1'][()]
                values['weights'] = data['W'][()]

            else:
                values['energy_injected'] = np.append(values['energy_injected'], data['E0'][()])
                values['energy_true'] = np.append(values['energy_true'], data['E'][()])
                values['t_delay'] = np.append(values['t_delay'], data['dt'][()])
                values['lon'] = np.append(values['lon'], data['Protsph'][1, :])
                values['lat'] = np.append(values['lat'], data['Protsph'][2, :])
                values['id_obs'] = np.append(values['id_obs'], data['ID'][()])
                values['id_parent'] = np.append(values['id_parent'], data['ID1'][()])
                values['weights'] = np.append(values['weights'], data['W'][()])

            hfile.close()

        edges['energy_injected'] *= u.eV
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

        elif isinstance(lightcurve, Iterable):
            edges['t_delay'] = np.array(lightcurve)

        else:
            first_bin = np.max([tmin, 1.])
            edges['t_delay'] =np.insert(np.logspace(np.log10(first_bin), 7, 8),
                                        0,
                                        insert_tmin)

            if tmax_data > edges['t_delay'].max():
                edges['t_delay'] = np.append(edges['t_delay'], np.max(tmax_data))

        edges['t_delay'] *= u.yr
        logging.info("Time bins: {0}".format(edges['t_delay']))
        # observed energy bins
        if isinstance(ebins, int):
            edges['energy_true'] = np.logspace(np.log10(values['energy_true'].min()),
                                          np.log10(values['energy_true'].max()),
                                          ebins) * u.eV

        elif isinstance(ebins, list) or isinstance(ebins, tuple):
            edges['energy_true'] = np.array(ebins)

        elif isinstance(ebins, np.ndarray):
            edges['energy_true'] = ebins

        # phi
        if int(gpv.split('.')[0]) < 1:
            binsz *= u.deg / u.pixel  # only necessary for gpv < 1.

        if isinstance(width, u.Quantity):
            width = width.to("deg")
        else:
            width *= u.deg
        nbins = np.ceil(2. * width / 2. / binsz).astype(np.int)
        edges['lon'] = np.linspace(-width.value / 2.,
                                   width.value / 2.,
                                   nbins.value + 1) * width.unit
        # theta
        edges['lat'] = np.linspace(90. - width.value / 2.,
                                   90. + width.value / 2.,
                                   nbins.value + 1) * width.unit
        # create a numpy histogram for the cascade
        # and potentially for the source
        # first make sure ordering is correct
        # needs to be in order t_delay, energy_injected, energy, lon, lat
        keys = ['t_delay', 'energy_injected', 'energy_true', 'lon', 'lat']
        edges = OrderedDict((k, edges[k]) for k in keys)
        return binsz, config, edges, n_injected_particles, values, width

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
        if isinstance(id_detection, Iterable):
            mc = np.zeros(values['id_obs'].size, dtype=np.bool)
            for idd in id_detection:
                mc |= (values['id_obs'] == idd)
            mc &= values['id_parent'] != id_injected

        else:
            mc = (values['id_obs'] == id_detection) & \
                 (values['id_parent'] != id_injected)
        # build data cube for cascade
        if np.sum(mc):
            data_casc = np.array([values[k][mc] for k in edges.keys()])
            # build the histogram
            logging.info("Building the cascade histogram ...")
            logging.info("Bin shapes: {0}".format([v.shape for v in edges.values()]))
            hist_casc, _ = np.histogramdd(data_casc.T, bins=[e.value for e in edges.values()],
                                          weights=values['weights'][mc])
            #print ("used weights", values['weights'][mc])
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
                keys = ['energy_injected', 'energy_true']
                data_primary = np.array([values[k][mi] for k in keys])
                if isinstance(config, dict):
                    # TODO: accounted for redshift, but is it correct?
                    # edges['e_true'] contain the injected energies
                    # edges['e_true'] / ( 1 + z) are the observed energies
                    hist_prim, edges_prim = np.histogramdd(data_primary.T,
                                                           bins=(edges['energy_injected'].value,
                                                                 edges['energy_injected'].value /
                                                                 (1. + config['Source']['z'])),
                                                           weights=values['weights'][mi]
                                                           )
                else:
                    hist_prim, edges_prim = np.histogramdd(data_primary.T,
                                                           bins=(edges['energy_injected'], edges['energy']),
                                                           weights=values['weights'][mi]
                                                           )
                edges_prim = [e * edges['energy_injected'].unit for e in edges_prim]
            else:
                logging.warning("no events pass primary spectrum criterion")
        return hist_casc, hist_prim, edges_prim

    def reset_casc(self):
        self._tmax = self._m.geom.axes['t_delay'].edges.max()
        self._casc = self._m.sum_over_axes(['t_delay'], keepdims=False)
        self._casc_obs = self._casc.sum_over_axes(['energy_injected'], keepdims=False)
        self._weights = np.ones_like(self._m.geom.axes['energy_injected'].center.value) * \
                        u.dimensionless_unscaled

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
        cen = self._casc.geom.axes['energy_injected'].center
        f = injspec(cen)
        lumi_iso = simps(f.value * cen.value, cen.value) * f.unit * cen.unit**2.
        if self._z is not None:
            lumi_iso *= cosmo.luminosity_distance(self._z).to('cm')**2. * 4. * np.pi
        else:
            return ValueError("Could not compute luminosity since redshift not provided.")
        lumi_iso *= doppler ** -4.
        return lumi_iso.to('erg s-1')

    def _compute_spectral_weights(self, injspec, **kwargs):
        """
        Set weights to compute cascade for an arbitrary spectrum.
        Spectrum should take energies in eV and return flux units in terms of eV.

        :param injspec: function pointer
            function that takes energy as Quantity and returns flux per energy

        :param kwargs: dict
            additional parameters passed to injspec

        :return:
        """

        # flux of new injected spectrum integrated in
        # bins of injected spectrum
        # as update for weights
        f = injspec(self._einj, **kwargs)

        target_unit = self._energy_injected.unit.to_string()

        # make sure that the right energy unit is used
        funit_split = f.unit.to_string().split('/')

        # Is f in E^2 dN / dE?
        if not "1" in funit_split[0]:
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

    def apply_spectral_weights(self, injspec, smooth=False, force_recompute=False, **kwargs):
        """
        Apply weights to compute cascade for an arbitrary spectrum

        :param injspec: function pointer
            function that takes energy as Quantity and returns flux per energy

        :param smooth: bool
            if True, apply adaptive smoothing to the cascade in each energy bin

        :param kwargs: dict
            additional parameters passed to injspec
        :return:
        """
        weights = self._compute_spectral_weights(injspec, **kwargs)
        # weights did not change, return
        if not self._weights.unit == u.dimensionless_unscaled and not force_recompute \
                and np.all(np.equal(weights, self._weights)):
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
                (self._casc * self._weights[:, np.newaxis, np.newaxis, np.newaxis]).sum_over_axes(['energy_injected'],
                                                                                                  keepdims=False)
            self._casc_obs /= self._casc_obs_bin_volume

        if smooth:
            self._asmooth.input_map = self._casc_obs
            res = self._asmooth.smooth()
            self._casc_obs = res["map"]

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

        add_primary: bool
            add the primary spectrum to the central pixel

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
                energy_obs_frame=self._casc_obs.geom.axes['energy_true'].center
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

    def export_casc_obs_to_fits(self, filename,
                                target_energy_unit='MeV',
                                target_flux_unit='MeV-1 s-1 cm-2 sr-1',
                                overwrite=True,
                                conv='fgst-template',
                                add_primary=False,
                                hdu_bands='ENERGIES',
                                extra_header_dict=None
                                ):
        """
        Export the current observed cascade flux to a fits file,
        compatible with a Fermi analysis.

        :param filename: str,
            path the fits file
        :param target_energy_unit:
            The target unit for the energy axis. Default: MeV
        :param target_flux_unit:
            The target unit for the flux. Default: MeV-1 s-1 cm-2 sr-1
        :param overwrite: bool
            Overwrite existing fits file
        :param conv: str
            {'gadf', 'fgst-ccube','fgst-template'}
            FITS format convention. Default: fgst-template
        :param add_primary: bool
            Add the primary spectrum to the central pixel. Default: False
        :param hdu_bands: str
            Name of the fits extension containing the energies. Default: ENERGIES
        :param extra_header_dict: dict
            Dictionary with additional key words for the header["META"] key word.
        :return:
        """
        if add_primary:
            map_export = self.add_primary_to_casc()
        else:
            map_export = self._casc_obs.copy()

        # force values smaller than zero to zero
        m = map_export.data <= 0.
        map_export.data[m] = 1e-40

        # create a new geometry with same WCS
        # but energy unit in MeV
        # since there does not seem to be an easy way to simply 
        # change the unit of the energy axis
        wcs = wcs = map_export.geom.drop("energy_true")  # get the spatial wcs
        # create the new energy axis in MeV
        energy_true_MeV = MapAxis.from_energy_edges(map_export.geom.axes["energy_true"].edges.to("MeV"),
                                                    name="energy")
        # set up the new geometry
        geom_new = WcsGeom.create(binsz=np.squeeze(wcs.width / wcs.npix),
                                  width=np.squeeze(wcs.width),
                                  skydir=wcs.center_skydir,
                                  frame=wcs.frame,
                                  axes=[energy_true_MeV])

        # create the new map and fill it
        wcs_map_export= WcsMap.from_geom(geom_new,
                                         data=map_export.quantity.to(target_flux_unit).value,
                                         unit=target_flux_unit)

        # export new map
        hdu_list = wcs_map_export.to_hdulist(hdu_bands=hdu_bands, format="fgst-template")

        if extra_header_dict is not None:
            if 'META' in hdu_list[0].header.keys():
                d = yaml.safe_load(hdu_list[0].header['META'])
                d.update(extra_header_dict)
            else:
                d = extra_header_dict
            hdu_list[0].header['META'] = json.dumps(d)

        hdu_list.writeto(filename, overwrite=overwrite)
        # free up memory
        del wcs_map_export, map_export

    def interpolate_spectrum(self,
                             radius=None,
                             on_region=None,
                             energy_unit="GeV",
                             dNdE_unit="TeV-1 cm-2 s-1",
                             **kwargs
                             ):
        """
        Compute a spline for spectral interpolation in log-log representation

        Parameters
        ----------
        on_region: extraction region or None
            region in which cascade is contribution is summed up
        radius: str or None
            if string, should be the angle of circular extraction region compatible with Angle, e.g., "0.1 deg".
            Will overwrite on_region.

        Returns
        -------
        tuple with spline and energies used for interpolation
        """
        kwargs.setdefault("k", 2)
        kwargs.setdefault("s", 1e-4)
        kwargs.setdefault("ext", r"extrapolate")

        if radius is not None:
            on_region = CircleSkyRegion(self._casc_obs.geom.center_skydir,
                                        radius=Angle(radius))

        spec_halo = self.get_obs_spectrum(
            region=on_region
        )

        spec_tot = self.get_obs_spectrum(
            region=on_region,
            add_primary=True
        )

        energy_halo = spec_halo.geom.axes['energy_true']
        energy_tot = spec_tot.geom.axes['energy_true']

        flux_unit_conversion = spec_halo.quantity.unit.to(dNdE_unit)

        x = np.log10(energy_halo.center.to(energy_unit).value)
        y = (spec_halo.data[:, 0, 0] * flux_unit_conversion)
        y[y == 0.] = 1e-60
        y = np.log10(y)

        spline = UnivariateSpline(x, y, **kwargs)

        return spline, energy_halo.center.to(energy_unit)

    def integrate_casc_spec(self,
                            energy_edges,
                            power=0,
                            x_steps=100,
                            radius=None,
                            on_region=None,
                            energy_unit="GeV",
                            dNdE_unit="TeV-1 cm-2 s-1",
                            **kwargs
                            ):
        """
        Integrate the cascade spectrum between energy edges from spline interpolation

        Parameters
        ----------
        :param radius:
        :param on_region:
        :param energy_unit:
        :param dNdE_unit:
        :param kwargs:

        Return
        ------
        The integral within energy edges
        """
        spline, energies = self.interpolate_spectrum(radius=radius,
                                                     energy_unit=energy_unit,
                                                     dNdE_unit=dNdE_unit,
                                                     **kwargs)

        integral = np.zeros(energy_edges.size - 1)
        for i, x in enumerate(energy_edges[:-1].to(energy_unit).value):
            x_ip1 = energy_edges[i+1].to(energy_unit).value
            x_array = np.logspace(np.log10(x), np.log10(x_ip1), x_steps)
            y = 10.**spline(np.log10(x_array)) * np.power(x_array, power)
            integral[i] = simps(y * x_array, np.log(x_array))

        integral_unit = u.Unit(dNdE_unit) * u.Unit(energy_unit) ** (1. + power)

        return integral * integral_unit


    def plot_spectrum(self,
                      radius=None,
                      on_region=None,
                      energy_unit="GeV",
                      E2dNdE_unit="TeV cm-2 s-1",
                      kwargs_casc={},
                      kwargs_tot={},
                      kwargs_prim={},
                      plot_errorbar=True,
                      ax=None,
                      fig=None):
        """
        Plot the cascade spectrum


        :param on_region: extraction region or None
            region in which cascade is contribution is summed up
        :param radius: str or None
            if string, should be the angle of circular extraction region compatible with Angle, e.g., "0.1 deg".
            Will overwrite on_region.
        :return:
        """
        import matplotlib.pyplot as plt

        plot_casc = kwargs_casc.pop("plot", True)
        plot_prim = kwargs_prim.pop("plot", True)
        plot_tot = kwargs_tot.pop("plot", True)

        kwargs_casc.setdefault("label", r"Cascade $\gamma$-ray spectrum")
        kwargs_prim.setdefault("label", r"Primary $\gamma$-ray spectrum")
        kwargs_tot.setdefault("label", r"Total $\gamma$-ray spectrum")

        kwargs_casc.setdefault("marker", ".")
        kwargs_prim.setdefault("marker", ".")
        kwargs_tot.setdefault("marker", ".")

        if fig is None:
            fig = plt.figure(figsize=(6,4))
        if ax is None:
            ax = fig.add_subplot(111)

        if radius is not None:
            on_region = CircleSkyRegion(self._casc_obs.geom.center_skydir,
                                        radius=Angle(radius))

        spec_halo = self.get_obs_spectrum(
            region=on_region
            )

        spec_tot = self.get_obs_spectrum(
            region=on_region,
            add_primary=True
            )

        energy_halo = spec_halo.geom.axes['energy_true']
        energy_tot = spec_tot.geom.axes['energy_true']

        flux_unit_conversion = (spec_halo.quantity.unit * energy_halo.unit ** 2.).to(E2dNdE_unit)

        # plot cascade
        if plot_casc:
            if plot_errorbar:
                ax.errorbar(energy_halo.center.to(energy_unit).value,
                            spec_halo.data[:, 0, 0] * energy_halo.center.value ** 2. * flux_unit_conversion,
                            xerr=energy_halo.bin_width.to(energy_unit).value / 2.,
                            **kwargs_casc
                            )
            else:
                ax.plot(energy_halo.center.to(energy_unit).value,
                        spec_halo.data[:, 0, 0] * energy_halo.center.value ** 2. * flux_unit_conversion,
                        **kwargs_casc
                        )

        if plot_prim and self._primary is not None:
            if plot_errorbar:
                ax.errorbar(self._primary.energy_obs_frame.center.to(energy_unit).value,
                            self._primary.get_obs_spectrum().value * self._primary.energy_obs_frame.center.value ** 2. *
                            flux_unit_conversion,
                            xerr=self._primary.energy_obs_frame.bin_width.to(energy_unit).value / 2.,
                            **kwargs_prim
                            )
            else:
                ax.plot(self._primary.energy_gal_frame.center.to(energy_unit).value,
                        self._primary.get_obs_spectrum().value * self._primary.energy_gal_frame.center.value ** 2. *
                        flux_unit_conversion,
                        **kwargs_prim
                        )

        if plot_tot and self._primary is not None:
            if plot_errorbar:
                ax.errorbar(energy_tot.center.to(energy_unit).value,
                            spec_tot.data[:, 0, 0] * energy_tot.center.value ** 2. * flux_unit_conversion,
                            xerr=energy_tot.bin_width.to(energy_unit).value / 2.,
                            **kwargs_tot
                            )
            else:
                ax.plot(energy_tot.center.to(energy_unit).value,
                        spec_tot.data[:, 0, 0] * energy_tot.center.value ** 2. * flux_unit_conversion,
                        **kwargs_tot
                        )

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel("Energy ({0:s})".format(energy_unit))
        ax.set_ylabel("$E^2 dN/dE$ ({0:s})".format(E2dNdE_unit))

        return fig, ax


class ASmooth(object):
    """
    Adaptively smooth an image.

    Achieves a roughly constant significance of features across the whole image.

    Adopted for Cascade model maps.

    Algorithm based on https://ui.adsabs.harvard.edu/abs/2006MNRAS.368...65E
    """

    def __init__(self, input_map, kernel=Tophat2DKernel, threshold=4., steps=50):


        # this should be 3. / 8. for a Tophat kernel according to ASmooth paper
        # but 1./8. gives better results for my case
        self._result = {}
        self._max_scale_factor = 1. / 8.

        self._threshold = threshold
        self._steps = steps
        self._scales = None
        self._input_map = input_map
        self._kernel = kernel

        self._set_scale()

    def _set_scale(self):
        min_angular_scale = self._input_map.geom.pixel_scales.min().to("deg")
        width = self._input_map.geom.width.min().to("deg")
        self._scales = np.linspace(min_angular_scale.value, width.value / 2. * self._max_scale_factor) * u.deg

    @property
    def input_map(self):
        return self._input_map

    @property
    def threshold(self):
        return self._threshold

    @property
    def kernel(self):
        return self._kernel

    @property
    def steps(self):
        return self._steps

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @input_map.setter
    def input_map(self, input_map):
        self._input_map = input_map
        self._set_scale()

    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @steps.setter
    def steps(self, steps):
        self._steps = steps

    def kernels(self, pixel_scale):
        """
        Ring kernels according to the specified method.

        Parameters
        ----------
        pixel_scale : `~astropy.coordinates.Angle`
            Sky image pixel scale

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Kernel`
        """
        # kernels operate in pixel space
        scales = self._scales.to_value("deg") / Angle(pixel_scale).deg
        logging.debug("kernel scales: {0}".format(scales))

        kernels = []
        for scale in scales:  # .value:
            kernel = self._kernel(scale, mode="oversample")
            # TODO: check if normalizing here makes sense
            kernel.normalize("peak")
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _significance(counts):
        """
        This should be the significance according to formula (5) in asmooth paper.
        Since we're dealing with a pure simulation here, we opt for
        a different definition, namely the counts above the mean value.
        """
        #return (counts - background) / np.sqrt(counts + background)
        mean = counts.mean()
        if not mean:
            mean = 1e-50
        return counts / mean

    def smooth(self, axis_name='energy_true'):
        """
        Run adaptive smoothing on input Map.

        Parameters
        ----------
        axis_name: str
            Name of axis that will be looped over

        Returns
        -------
        images : dict of `~gammapy.maps.WcsNDMap`
            Smoothed images; keys are:
                * 'map'
                * 'scales'
                * 'significance'.
        """
        pixel_scale = self._input_map.geom.pixel_scales.mean()

        # for each chosen scale, generate a kernel
        kernels = self.kernels(pixel_scale)

        for k in ["map", "scale", "significance"]:
            self._result[k] = self._input_map.copy()

        if int(gpv.split('.')[0]) < 1:
            self._result["scale"].unit = "deg"
            self._result["significance"].unit = ""

        cubes = {}

        # loop over energy bins
        for i in range(self._input_map.geom.axes[axis_name].center.size):
            # for each of the chosen scales,
            # this does the convolution
            cubes["map"] = scale_cube(self._input_map.slice_by_idx({axis_name: i}).data, kernels)

            # Compute significance for each convolution scale
            cubes["significance"] = self._significance(cubes["map"])
            smoothed = self._reduce_cubes(cubes, kernels)

            # set remaining pixels with significance < threshold to constant value
            for key in ["map", "scale", "significance"]:
                data = smoothed[key]

                # set remaining pixels with significance < threshold to mean value
                if key in ["map"]:
                    mask = np.isnan(data)
                    data[mask] = 0.  # np.mean(locals()[key].data[mask])
                    self._result[key].data[i] = data
                else:
                    self._result[key].data[i] = data

        return self._result

    def _reduce_cubes(self, cubes, kernels):
        """
        Combine scale cube to image.

        Parameters
        ----------
        cubes : dict
            Data cubes
        """
        shape = cubes["map"].shape[:2]
        smoothed = {}

        # Init smoothed data arrays
        for key in ["map", "scale", "significance"]:
            smoothed[key] = np.tile(np.nan, shape)

        # loop over all kernels to adaptively smooth image
        for idx, scale in enumerate(self._scales):
            # slice out 2D image at index idx out of cube
            slice_ = np.s_[:, :, idx]

            mask = np.isnan(smoothed["map"])
            mask = (cubes["significance"][slice_] > self._threshold) & mask

            smoothed["scale"][mask] = scale
            smoothed["significance"][mask] = cubes["significance"][slice_][mask]

            # renormalize smoothed data arrays
            norm = kernels[idx].array.sum()
            for key in ["map"]:
                smoothed[key][mask] = cubes[key][slice_][mask] / norm

        return smoothed

    def diagnostic_plots(self, fig=None, fig2=None, cmap='cubehelix_r', slice="sum", axis='energy_true', cutout=None,
                         width=0.1 * u.deg):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if fig is None:
            fig = plt.figure(1, figsize=(10, 10))
        if fig2 is None:
            fig2 = plt.figure(2, figsize=(6, 4))

        if cutout is None:
            cutout = self._input_map.geom.width

        c = self._input_map.geom.center_skydir

        if isinstance(slice, int):
            sig = self._result["significance"].slice_by_idx({axis: slice})
            scale = self._result["scale"].slice_by_idx({axis: slice})
            input_map = self._input_map.slice_by_idx({axis: slice})
            smooth = self._result["map"].slice_by_idx({axis: slice})

            ax00 = fig.add_subplot(221, projection=input_map.cutout(c, width=cutout).geom.wcs)
            ax01 = fig.add_subplot(222, projection=smooth.cutout(c, width=cutout).geom.wcs)
            ax10 = fig.add_subplot(223, projection=sig.cutout(c, width=cutout).geom.wcs)
            ax11 = fig.add_subplot(224, projection=scale.cutout(c, width=cutout).geom.wcs)

            sig.cutout(c, width=cutout).plot(
                ax=ax10, add_cbar=True, stretch='log', cmap=cmap)
            scale.cutout(c, width=cutout).plot(
                ax=ax11, add_cbar=True, stretch='log', cmap=cmap)

        elif isinstance(slice, str) and slice == 'sum':
            input_map = self._input_map.sum_over_axes([axis], keepdims=False)
            smooth = self._result["map"].sum_over_axes([axis], keepdims=False)

            ax00 = fig.add_subplot(221, projection=input_map.cutout(c, width=cutout).geom.wcs)
            ax01 = fig.add_subplot(222, projection=smooth.cutout(c, width=cutout).geom.wcs)

        else:
            raise ValueError("Value of slice keyword not understood")

        input_map.cutout(c, width=cutout).plot(
            ax=ax00, add_cbar=True, stretch='log', cmap=cmap)
        smooth.cutout(c, width=cutout).plot(
            ax=ax01, add_cbar=True, stretch='log', cmap=cmap)

        for a in [ax00, ax01]:
            if a == ax00:
                cmap = plt.cm.Blues
            else:
                cmap = plt.cm.Reds

            r1 = Rectangle((c.ra.value - self._input_map.geom.width[0, 0].value / 2.,
                            c.dec.value - width.value / 2.),
                           self._input_map.geom.width[0, 0].value, width.value,
                           edgecolor=cmap(0.9), facecolor='none', ls='--',
                           transform=a.get_transform('fk5'))

            r2 = Rectangle((c.ra.value - width.value / 2.,
                            c.dec.value - self._input_map.geom.width[1, 0].value / 2.),
                           width.value, self._input_map.geom.width[0, 0].value,
                           edgecolor=cmap(0.7), facecolor='none', ls='--',
                           transform=a.get_transform('fk5'))
#
            a.add_patch(r1)
            a.add_patch(r2)

        # plot lon and lat profiles
        ax20 = fig2.add_subplot(211)
        ax21 = fig2.add_subplot(212)

        ax20.semilogy(input_map.cutout(c, width=(self._input_map.geom.width[0, 0], width)).data.sum(axis=0),
                      color=plt.cm.Blues(0.9))
        ax20.semilogy(smooth.cutout(c, width=(self._input_map.geom.width[0, 0], width)).data.sum(axis=0),
                      color=plt.cm.Reds(0.9), ls='--')

        ax21.semilogy(input_map.cutout(c, width=(width, self._input_map.geom.width[1, 0])).data.sum(axis=1),
                      color=plt.cm.Blues(0.7))
        ax21.semilogy(smooth.cutout(c, width=(width, self._input_map.geom.width[1, 0])).data.sum(axis=1),
                      color=plt.cm.Reds(0.7), ls='--')

        return fig, fig2


