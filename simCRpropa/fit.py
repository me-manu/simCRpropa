import numpy as np
import iminuit as minuit
import time
import functools
import logging
from astropy.table import Table
from astropy import units as u
from scipy import optimize as op
from collections import OrderedDict
from ebltable.tau_from_model import OptDepth
from copy import deepcopy
from astropy.coordinates import Angle
from regions import CircleSkyRegion
from scipy.interpolate import interp1d


def set_default(func=None, passed_kwargs={}):
    """
    Read in default keywords of the simulation and pass to function
    """
    if func is None:
        return functools.partial(set_default, passed_kwargs=passed_kwargs)

    @functools.wraps(func)
    def init(*args, **kwargs):
        for k in passed_kwargs.keys():
            kwargs.setdefault(k,passed_kwargs[k])
        return func(*args, **kwargs)
    return init


def pl_exp_cut(x, mask=None, **params):
    """
    Power law with exponential cut off and energy scaling factor

    Parameters
    ----------
    x: array-like
        Energy values

    mask: array-like
        mask for energies to which additional energy scale is applied

    Returns
    -------
    Array with flux values
    """
    energy_scale = np.ones(x.shape)
    if mask is not None:
        # E -> E * (1 + s)
        energy_scale[mask] += params['Energy_Scale']
    else:
        # apply to all energies
        energy_scale += params['Energy_Scale']

    if isinstance(x, u.Quantity):
        energy_scale *= u.dimensionless_unscaled

    x_scaled = x * energy_scale
    result = params["Prefactor"] * np.power(x_scaled / params["Scale"], params["Index"])
    result *= np.exp(-x_scaled / params["Cutoff"])
    return result


def pl_exp_cut_low_high(x, **params):
    """
    Power law with exponential cut-off at low and high energies

    Parameters
    ----------
    x: array-like
        Energy values

    Returns
    -------
    Array with flux values
    """
    result = (x / params['Scale_CR']) ** (-params['Index_CR'])
    result *= params['Prefactor_CR']
    result *= np.exp(-x / params['Emax_CR']) * np.exp(-params['Emin_CR'] / x)
    return result


def log_parabola(x, **params):
    """
    Log parabola

    Parameters
    ----------
    x: array-like
        Energy values

    Returns
    -------
    Array with flux values
    """
    result = (x / params['Scale']) ** (-params['Index'] - params['Curvature'] * np.log(x / params['Scale']))
    result *= params['Prefactor']
    return result


minuit_def = {
    'verbosity': 0,
    'int_steps': 1e-4,
    'strategy': 2,
    'tol': 1e-5,
    'up': 1.,
    'max_tol_increase': 3000.,
    'tol_increase': 1000.,
    'ncall': 5000,
    'pedantic': True,
    'precision': None,
    'scipy': True,
    'pinit': {'Prefactor': -10.,
              'Index': -3.,
              'Scale': 1000.,
              'Energy_Scale': 1.,
              'Cutoff': 1.},
    'fix': {'Prefactor': False,
            'Index': False,
            'Cutoff': False,
            'Energy_Scale': False,
            'Scale': True},
    'limits': {'Prefactor' : [-20, -5],
               'Index': [-5, 5],
               'Cutoff': [0.1, 10.],
               'Energy_Scale': [0.5, 1.5],
               'Scale': [16., 18.]},
    'islog': {'Prefactor' : True,
              'Index': False,
              'Cutoff': False,
              'Energy_Scale': False,
              'Scale': False},
}



class FitIACTFermi(object):
    """
    Class to perform fit of intrinsic spectrum
    on IACT data and Fermi-LAT Likelihood cube
    to derive limits on the intergalactic magnetic field
    """
    def __init__(self, x, y, dy, z, dx=None, x_min=None, x_max=None,
                 llh_fermi_interp=None, casc=None,
                 ebl_model='dominguez',
                 interp_casc=True,
                 on_region_radius=0.2):
        """
        Initialize the class

        Parameters
        ----------
        x: array-like
            Energy values in TeV for IACT measurement

        y: array-like
            Flux values in TeV for IACT measurement in dN / dE format in units of (TeV s cm^2)^-1

        dy: array-like
            Errors on flux

        z: float
            source redshift

        dx: array-like or None
            Bin width in TeV

        llh_fermi_interp: interpolation function
            Function that receives spectral parameters as input and returns the Fermi-LAT
            likelhood

        casc: `~cascmaps.CascMap`
            Cascade map container

        ebl_model: str
            EBL model identifier

        on_region_radius: float
            assumed size for ON region in degrees

        interp_casc: bool
            if True, use 1D cubic interpolation to calculate
            cascade contribution to IACT spectrum
        """

        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._x_min = x_min
        self._x_max = x_max

        self._llh_fermi_interp = llh_fermi_interp
        self._casc = casc
        self._par_names = None
        self._par_islog = None
        self._cov_inv = None
        self._minimize_f = None
        self._m = None
        self._res = None
        self._z = z
        self._y_pred = None

        self._tau = OptDepth.readmodel(model=ebl_model)
        self._atten = np.exp(-self._tau.opt_depth(self._z, self._x))

        self._on_region_rad = Angle(on_region_radius, unit="deg")
        self._on_region = None
        self._llh_fermi = None

        self._cr_spec = None
        self._int_spec = None
        self._interp_casc = interp_casc

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def dx(self):
        return self._dx

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def z(self):
        return self._z

    @property
    def dy(self):
        return self._dy

    @property
    def llh_fermi_interp(self):
        return self._llh_fermi_interp

    @property
    def llh(self):
        return self._llh

    @property
    def llh_fermi(self):
        return self._llh_fermi

    @property
    def casc(self):
        return self._casc

    @property
    def interp_casc(self):
        return self._interp_casc

    @llh_fermi_interp.setter
    def llh_fermi_interp(self, llh_fermi_interp):
        self._llh_fermi_interp = llh_fermi_interp

    @casc.setter
    def casc(self, casc):
        self._casc = casc
        if casc is not None:
            self._on_region = CircleSkyRegion(casc.casc_obs.geom.center_skydir,
                                  radius=Angle(self._on_region_rad))

    @interp_casc.setter
    def interp_casc(self, interp_casc):
        self._interp_casc = interp_casc

    @staticmethod
    def read_iact_fits_file(iact_file, sed_name, flux_unit='TeV-1 s-1 m-2'):
        """
        Initialize the class with IACT spectrum from fits file

        Parameters
        ----------
        iact_file: str
            path to fits file

        sed_name: str
            name of SED to use

        flux_unit: str
            unit of flux in SED file
        """

        data = Table.read(iact_file)
        m = data["SOURCE_FULL"] == sed_name
        if not np.sum(m):
            raise ValueError("{0:s} not in list: {1}".format(sed_name, data["SOURCE_FULL"]))
        x = data["E_REF"][m].data
        y = data["NORM"][m].data * u.Unit(flux_unit).to("TeV-1 s-1 cm-2")
        dy = 0.5 * (data["NORM_ERRN"][m] + data["NORM_ERRP"][m]).data * u.Unit(flux_unit).to("TeV-1 s-1 cm-2")

        mask = np.isfinite(x)
        log_xerr = np.insert(np.diff(np.log10(x[mask])), 0, np.diff(np.log10(x[mask]))[0])
        log_x_edges = np.append(np.log10(x[mask]) - log_xerr / 2.,
                                np.log10(x[mask][-1]) + log_xerr[-1] / 2.)
        x_edges = np.power(10., log_x_edges)
        x_min = x_edges[:-1]
        x_max = x_edges[1:]
        x_cen = np.sqrt(x_edges[1:] * x_edges[:-1])

        return FitIACTFermi(x_cen, y[mask], dy[mask],
                            data["REDSHIFT"][m].data[0],
                            x_min=x_min, x_max=x_max)

    @staticmethod
    def read_magic_fits_file(magic_file, redshift, flux_unit='TeV-1 s-1 cm-2', hdu=2, energy_unit='TeV'):
        """
        Read in fits file obtained from the MAGIC website
        """
        sed = Table.read(magic_file, hdu=hdu)
        x_cen = sed['energy'].to(energy_unit)
        dx = sed['Denergy'].to(energy_unit)

        sed['flux'].unit = u.Unit(sed['flux'].unit.to_string().replace("ph", ""))
        sed['Dflux'].unit = sed['flux'].unit

        y = (sed['flux'] / x_cen.to('TeV') ** 2.).to(flux_unit)
        dy = (sed['Dflux'] / x_cen.to('TeV') ** 2.).to(flux_unit)
        x_edges = np.append(x_cen - dx / 2., x_cen[-1] + dx[-1] / 2.)
        x_min = x_edges[:-1]
        x_max = x_edges[1:]

        return FitIACTFermi(x_cen.to("TeV").value, y.value, dy.value,
                            redshift,
                            x_min=x_min.to("TeV").value, x_max=x_max.to("TeV").value)

    def add_fermi_sed_points(self, sed, ts_thr=9., energy_unit="TeV", flux_unit="TeV-1 cm-2 s-1"):
        """
        Add SED points from Fermi-LAT analysis

        Parameters
        ----------
        sed: dict
            dictionary with Fermi-LAT SED generated with fermipy

        ts_thr: float
            Threshold TS value, only energy bins above this threshold will be used

        energy_unit: str
            Target energy unit

        flux_unit: str
            Target flux unit for dN/dE entries

        Notes
        -----
        energy edges might not work any more after
        this
        """

        m_ts = sed['ts'] >= ts_thr
        self._x = np.insert(self._x, 0, sed['e_ref'][m_ts] * u.MeV.to(energy_unit))
        self._x_min = np.insert(self._x_min, 0, sed['e_min'][m_ts] * u.MeV.to(energy_unit))
        self._x_max = np.insert(self._x_max, 0, sed['e_max'][m_ts] * u.MeV.to(energy_unit))
        self._y = np.insert(self._y, 0, sed['dnde'][m_ts] * u.Unit("MeV-1 s-1 cm-2").to(flux_unit))
        self._dy = np.insert(self._dy, 0, sed['dnde_err'][m_ts] * u.Unit("MeV-1 s-1 cm-2").to(flux_unit))


        atten = np.exp(-self._tau.opt_depth(self._z, sed['e_ref'][m_ts] * u.MeV.to("TeV")))
        self._atten = np.insert(self._atten, 0, atten)

    def calc_likelihood(self, *args):
        return self.__calc_likelihood(*args)

    def __calc_likelihood(self, *args):
        """
        likelihood function passed to iMinuit
        """
        params = {}
        for i, p in enumerate(self._par_names):
            if self._par_islog[p]:
                params[p] = np.power(10., args[i])
            else:
                params[p] = args[i]
        return self.return_likelihood(params)

    def __wrap_likelihood(self, args):
        """
        likelihood function passed to scipy.optimize
        """
        params = {}
        for i, p in enumerate(self._par_names):
            if not self.fitarg['fix'][p]:
                if self._par_islog[p]:
                    params[p] = np.power(10., args[i])
                else:
                    params[p] = args[i]
            else:
                if self._par_islog[p]:
                    params[p] = np.power(10., self.fitarg['pinit'][p])
                else:
                    params[p] = self.fitarg['pinit'][p]
        return self.return_likelihood(params)

    def return_likelihood(self, params):
        """Calculate the log likelihood"""

        self._y_pred = self._int_spec(self._x, **params) * self._atten

        # Add the cascade
        if self._casc is not None:
            params_casc = deepcopy(params)

            # apply the weights
            if self._cr_spec is not None:
                # add units to the parameters where neccessary
                params_casc['Prefactor_CR'] *= u.Unit("TeV-1 cm-2 s-1")
                params_casc['Scale_CR'] *= u.Unit("eV").to("eV") * u.eV
                params_casc['Emin_CR'] *= u.Unit("eV").to("eV") * u.eV
                params_casc['Emax_CR'] *= u.Unit("eV").to("eV") * u.eV
                self._casc.apply_spectral_weights(lambda x: self._cr_spec(x, **params_casc),
                                                  smooth=True)

            else:
                # add units to the parameters where neccessary
                params_casc['Prefactor'] *= u.Unit("TeV-1 cm-2 s-1")
                params_casc['Scale'] *= u.Unit("TeV").to("eV") * u.eV
                params_casc['Cutoff'] *= u.Unit("TeV").to("eV") * u.eV
                self._casc.apply_spectral_weights(lambda x: self._int_spec(x, **params_casc),
                                                  smooth=True)

            # and get the flux in the ON region
            spec_halo = self._casc.get_obs_spectrum(
                region=self._on_region
            )
            # convert the units back
            flux_unit_conversion = (spec_halo.quantity.unit).to("TeV-1 cm-2 s-1")

            # either add directly if energy bins are the same or use 1D interpolation
            if self._interp_casc:
                m = spec_halo.data[:, 0, 0] > 0.
                if not np.sum(m):
                    raise ValueError("Predicted cascade flux is zero!")
                interp = interp1d(np.log(spec_halo.geom.get_axis_by_name('energy').center.to("TeV").value[m]),
                                  np.log(spec_halo.data[:, 0, 0][m] * flux_unit_conversion),
                                  fill_value='extrapolate', kind='cubic'
                                  )
                self._y_pred += np.exp(interp(np.log(self._x)))

            else:
                self._y_pred += spec_halo.data[:, 0, 0] * flux_unit_conversion

        if self._cov_inv is None:
            self._llh = -1. * ((self._y - self._y_pred) ** 2. / self._dy ** 2.).sum()
        else:
            self._llh = -1. * np.dot(self._y - self._y_pred, np.dot(self._cov_inv, self._y - self._y_pred))

        # add contribution from profile likelihood
        if self._llh_fermi_interp is not None:
            # change parameters to the values over which grid was interpolated
            params_llh = deepcopy(params)
            params_llh['Prefactor'] *= u.Unit("TeV-1 cm-2 s-1").to("MeV-1 cm-2 s-1")
            params_llh['Index'] *= -1.
            self._llh_fermi = 2. * self._llh_fermi_interp([params_llh['Cutoff'],
                                                           -1. * params_llh['Index'],
                                                           np.log10(params_llh['Prefactor'])])[0]
        else:
            self._llh_fermi = 0

        return -1. * (self._llh + self._llh_fermi)

    @set_default(passed_kwargs=minuit_def)
    def fill_fitarg(self, **kwargs):
        """
        Helper function to fill the dictionary for minuit fitting
        """
        # set the fit arguments
        fitarg = {}
        #fitarg.update(kwargs['pinit'])
        #for k in kwargs['limits'].keys():
        #    fitarg['limit_{0:s}'.format(k)] = kwargs['limits'][k]
        #    fitarg['fix_{0:s}'.format(k)] = kwargs['fix'][k]
        #    fitarg['error_{0:s}'.format(k)] = kwargs['pinit'][k] * kwargs['int_steps']
#
#        fitarg = OrderedDict(sorted(fitarg.items()))
        fitarg['pinit'] = kwargs['pinit']
        fitarg['limits'] = kwargs['limits']
        fitarg['fix'] = kwargs['fix']
        fitarg['error'] = OrderedDict()

        for k in kwargs['limits'].keys():
            fitarg['error'][k] = kwargs['pinit'][k] * kwargs['int_steps']

        # get the names of the parameters
        self._par_names = list(kwargs['pinit'].keys())
        self._par_islog = kwargs['islog']
        return fitarg

    @set_default(passed_kwargs=minuit_def)
    def run_migrad(self, fitarg, **kwargs):
        """
        Helper function to initialize migrad and run the fit.
        Initial parameters are estimated with scipy fit.
        """
        self.fitarg = fitarg
        kwargs['up'] = 1.


        logging.debug(self._par_names)
        logging.debug(self.__wrap_likelihood(list(fitarg['pinit'].values())))

        if kwargs['scipy']:
            self._res = op.minimize(self.__wrap_likelihood,
                                   list(fitarg['pinit'].values()),
                                   bounds=list(fitarg['limits'].values()),
                                   method='TNC',
                                   #method='Powell',
                                   options={'maxiter': kwargs['ncall']} #'xtol': 1e-20, 'eps' : 1e-20, 'disp': True}
                                   #tol=None, callback=None,
                                   #options={'disp': False, 'minfev': 0, 'scale': None,
                                   #'rescale': -1, 'offset': None, 'gtol': -1,
                                   #'eps': 1e-08, 'eta': -1, 'maxiter': kwargs['ncall'],
                                   #'maxCGit': -1, 'mesg_num': None, 'ftol': -1, 'xtol': -1, 'stepmx': 0,
                                   #'accuracy': 0}
                                   )
            logging.info(self._res)
            for i, k in enumerate(self._par_names):
                fitarg[k] = self._res.x[i]

            logging.debug(fitarg)

        cmd_string = "lambda {0}: self.__calcLikelihood({0})".format(
            (", ".join(self._par_names), ", ".join(self._par_names)))

        string_args = ", ".join(self._par_names)
        global f # needs to be global for eval to find it
        f = lambda *args: self.__calc_likelihood(*args)

        cmd_string = "lambda %s: f(%s)" % (string_args, string_args)
        logging.debug(cmd_string)

        # work around so that the parameters get names for minuit
        self._minimize_f = eval(cmd_string, globals(), locals())
        self._minimize_f.errordef = minuit.Minuit.LEAST_SQUARES

        self._m = minuit.Minuit(self._minimize_f,
                                #list(fitarg['pinit'].values()),
                                **fitarg['pinit'],
                                #names=self._par_names
                                )
#                               print_level=kwargs['verbosity'],
#                               errordef=kwargs['up'],
#                               pedantic=kwargs['pedantic'],
                               #**fitarg)

        for p in self._par_names:
            self._m.fixed[p] = fitarg['fix'][p]
            self._m.limits[p] = fitarg['limits'][p]
            self._m.errors[p] = fitarg['error'][p]

        self._m.tol = kwargs['tol']
        self._m.strategy = kwargs['strategy']

        logging.debug("tol {0:.2e}, strategy: {1:n}".format(
            self._m.tol, self._m.strategy.strategy))

        self._m.migrad(ncall=kwargs['ncall']) #, precision = kwargs['precision'])

    def __print_failed_fit(self):
        """print output if migrad failed"""
        if not self._m.valid:
            fmin = self._m.fmin
            logging.warning(
                '*** migrad minimum not valid! Printing output of get_fmin'
            )
            logging.warning(self._m.fmin)
            logging.warning('{0:s}:\t{1}'.format('*** has_accurate_covar',
                                                 fmin.has_accurate_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_covariance',
                                                 fmin.has_covariance))
            logging.warning('{0:s}:\t{1}'.format('*** has_made_posdef_covar',
                                                 fmin.has_made_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_posdef_covar',
                                                 fmin.has_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_reached_call_limit',
                                                 fmin.has_reached_call_limit))
            logging.warning('{0:s}:\t{1}'.format('*** has_valid_parameters',
                                                 fmin.has_valid_parameters))
            logging.warning('{0:s}:\t{1}'.format('*** hesse_failed',
                                                 fmin.hesse_failed))
            logging.warning('{0:s}:\t{1}'.format('*** is_above_max_edm',
                                                 fmin.is_above_max_edm))
            logging.warning('{0:s}:\t{1}'.format('*** is_valid',
                                                 fmin.is_valid))

    def __repeat_migrad(self, **kwargs):
        """Repeat fit if fit was above edm"""
        fmin = self._m.fmin
        if not self._m.valid and fmin.is_above_max_edm:
            logging.warning(
                'Migrad did not converge, is above max edm. Increasing tol.'
            )
            tol = self._m.tol
            self._m.tol *= self._m.edm / (self._m.tol * self._m.errordef) * kwargs['tol_increase']

            logging.info('New tolerance : {0}'.format(self._m.tol))
            if self._m.tol >= kwargs['max_tol_increase']:
                logging.warning(
                    'New tolerance to large for required precision'
                )
            else:
                self._m.migrad(
                    ncall=kwargs['ncall'])

                logging.info(
                    'Migrad status after second try: {0}'.format(
                        self._m.valid
                    )
                )
                self._m.tol = tol
        return

    @set_default(passed_kwargs=minuit_def)
    def fit(self, int_spec, cr_spec=None, minos=0., refit=True, **kwargs):
        """
        Fit an intrinsic spectrum

        Parameters
        ----------
        int_spec: function pointer
            function pointer to intrinsic gamma-ray spectrum that accepts energies in GeV and has the
            call signature f(ETeV, **parameters)

        cr_spec: function pointer
            function pointer to intrinsic spectrum that accepts energies in GeV and has the
            call signature f(ETeV, **parameters)

        kwargs
        ------
        pinit: dict
            initial guess for intrinsic spectral parameters

        fix: dict
            booleans for freezing parameters

        bounds: dict
            dict with list for each parameter with min and max value


        Returns
        -------
        tuple with likelihood profile for distance of
        gamma-ray emitting region
        """

        self._int_spec = lambda EGeV, **kwargs: int_spec(EGeV, **kwargs)

        if cr_spec is not None:
            self._cr_spec = lambda EGeV, **kwargs: cr_spec(EGeV, **kwargs)
        else:
            self._cr_spec = None

        fitarg = self.fill_fitarg(**kwargs)

        t1 = time.time()
        self.run_migrad(fitarg, **kwargs)

        try:
            self._m.hesse()
            logging.debug("Hesse matrix calculation finished")
        except RuntimeError as e:
            logging.warning(
                "*** Hesse matrix calculation failed: {0}".format(e)
            )

        logging.debug(self._m.fval)
        self.__repeat_migrad(**kwargs)
        logging.debug(self._m.fval)

        fmin = self._m.fmin

        if not fmin.hesse_failed:
            try:
                self._corr = self._m.np_matrix(correlation=True)
            except:
                self._corr = -1

        logging.debug(self._m.values)

        if self._m.valid and minos:
            for k in self._par_names:
                if kwargs['fix'][k]:
                    continue
                self._m.minos(k, minos)
            logging.debug("Minos finished")

        else:
            self.__print_failed_fit()

        logging.info('fit took: {0}s'.format(time.time() - t1))
        for k in self._par_names:
            if kwargs['fix'][k]:
                err = np.nan
            else:
                err = self._m.errors[k]
            logging.info('best fit {0:s}: {1:.5e} +/- {2:.5e}'.format(k, self._m.values[k], err))
