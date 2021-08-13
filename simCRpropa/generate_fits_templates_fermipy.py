import argparse
import yaml
import logging
import numpy as np
import glob
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.convolution import Tophat2DKernel, Gaussian2DKernel
from os import path
from copy import deepcopy
from fermiAnalysis.batchfarm import utils
from simCRpropa.cascmaps import CascMap, stack_results_lso

def pl2_to_pl(src_dict, scale=1000.):
    """Convert integral flux of PL2 to prefactor of PL"""
    index = src_dict['spectral_pars']['Index']['value']
    emin = src_dict['spectral_pars']['LowerLimit']['value']
    emax = src_dict['spectral_pars']['UpperLimit']['value']
    f = src_dict['spectral_pars']['Integral']['value']

    prefactor = f * (1. - index) 
    prefactor /= (emax ** (1. - index) - emin ** (1. - index))
    prefactor *= scale ** -index
    return prefactor

def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data

class GenerateFitsTemplates(object):
    def __init__(self, **kwargs):
        """
        Initialize the class
        """
        self._config = deepcopy(kwargs)
        self.__dict__.update(self._config)

        self.emcasc = self.Simulation['emcasc']

        for i, k in enumerate(['B', 'maxTurbScale']):
            if isinstance(self.Bfield[k], list):
                x = deepcopy(self.Bfield[k])
                self.Bfield[k] = x[0]
            elif isinstance(self.Bfield[k], float):
                x = [self.Bfield[k]]
            else:
                raise ValueError("{0:s} type not understood: {1}".format(
                    type(k, self.Bfield[k])))
            if not i:
                self._bList = x
            else:
                self._turbScaleList = x

        for i, k in enumerate(['th_jet', 'z']):
            if isinstance(self.Source[k], list):
                x = deepcopy(self.Source[k])
                self.Source[k] = x[0]
            elif isinstance(self.Source[k], float):
                x = [self.Source[k]]
            else:
                raise ValueError("{0:s} type not understood: {1}".format(
                    type(k, self.Source[k])))
            if not i:
                self._th_jetList= x
            else:
                self._zList = x

    def setOutput(self, idB=0, idL=0, it=0, iz=0):
        """Set output file and directory"""
        self.OutName = 'combined.hdf5'

        self.Source['th_jet'] = self._th_jetList[it]
        self.Source['z'] = self._zList[iz]

        # append options to file path
        self.FileIO['outdir'] = path.join(self.FileIO['basedir'],
                                'z{0[z]:.3f}'.format(self.Source))
        if self.Source.get('source_morphology', 'cone') == 'cone':
            self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                            'th_jet{0[th_jet]}/'.format(self.Source))
        elif self.Source.get('source_morphology', 'cone') == 'iso':
            self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                                              'iso/')
        elif self.Source.get('source_morphology', 'cone') == 'dir':
            self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                                                'dir/')
        else:
            raise ValueError("Chosen source morphology not supported.")
        self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                        'th_obs{0[obsAngle]}/'.format(self.Observer))
        self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                        'spec{0[useSpectrum]:n}/'.format(self.Source))

        self.Bfield['B'] = self._bList[idB]
        self.Bfield['maxTurbScale'] = self._turbScaleList[idL]

        if self.Bfield['type'] == 'turbulence':
            self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                'Bturb{0[B]:.2e}/q{0[turbIndex]:.2f}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield))
        elif self.Bfield['type'] =='cell':
            self.FileIO['outdir'] = path.join(self.FileIO['outdir'],
                'Bcell{0[B]:.2e}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield))
        else:
            raise ValueError("Bfield type must be either 'cell' or 'turbulence' not {0[type]}".format(self.Bfield))

        self.outputfile = str(path.join(self.FileIO['outdir'],self.OutName))
        logging.info("outdir: {0[outdir]:s}".format(self.FileIO))
        logging.info("outfile: {0:s}".format(self.outputfile))

    def generate_fits_templates(self,
                                fermipy_files,
                                select_z=None,
                                tmax=1e7,
                                theta_obs=0.,
                                #cov=2.,
                                cov=0.75,
                                #index_step=0.5,
                                index_step=0.075,
                                Ecut_TeV=np.arange(1.,13., 2.),
                                rotation=0.,
                                ts_thr=25., 
                                redshifts=None,
                                use_closest_z=True,
                                dry=False,
                                make_plots=False,
                                n_ebins_add=3,
                                select_src=None,
                                select_bfield=None,
                                overwrite=False):
        """
        Generate IGMF fits templates for sources analyzed with fermipy

        Parameters
        ----------
        fermipy_files: list
            list of npy files that are the result of an ROI fermipy fit

        select_z: float or None
            if not none, only generate templates for this redshift

        select_bfield: float or None
            if not none, only generate templates for this magnetic field

        theta_obs: float
            Angle between jet axis and line of sight in degrees

        tmax: float
            Maximum allowed delay time of cascade photons in years.

        cov: float
            determines the range of the spectral index loop
            through index + cov * error

        index_step: float
            increment for spectral index loop,
            cov_loop = np.arange(-cov, cov + index_step, index_step)

        Ecut_TeV: array-like
            assumed exponential cutoff energies in TeV

        rotation: float
            Angle in degrees by which cascade template is rotated

        ts_thr: float
            Only consider sources for which fermipy analysis gave ts value larger than this 
            value

        use_closest_z: bool
            if True, use template with redshift closest to source redshift, 
            otherwise, redshift has to match exactly

        redshifts: array-like
            list with source redshifts. If not given, it's extracted from the fermipy
            files. This only works if the sources were fitted with EBL absorption

        dry: bool
            only perform template generation if this is False.
            Useful for debugging.

        overwrite: bool
            Overwrite existing templates.

        n_ebins_add: int
            number of energy bins that will be added at low and high energy end
        """
        if make_plots:
            from myplot.spectrum import SEDPlotter
            import matplotlib.pyplot as plt

        # the assumed intrinsic spectrum:
        # power law with exponential cut off
        inj_spec = lambda E, **p : p['Prefactor'] * (E / p['Scale']) ** (-p['Index']) * \
                                   np.exp(-E / p['Cutoff'])

        # the steps for the index: Index + (Index_err) * cov_scale
        cov_scale = np.arange(-cov, cov + index_step, index_step)
        logging.info("Ecut_TeV: {0}".format(Ecut_TeV))
        logging.info("cov_scale: {0}".format(cov_scale))
        logging.info("{0:n} x {1:n} = {2:n} files will be generated for each source and B field config".format(
            Ecut_TeV.shape[0], cov_scale.shape[0], Ecut_TeV.shape[0] * cov_scale.shape[0]))

        for ifile, f in enumerate(fermipy_files):
            if not path.exists(f):
                logging.warning("{0:s} not found".format(f))
                continue

            d = np.load(f, allow_pickle=True, encoding="latin1").flat[0]
            src = d['config']['selection']['target']
            src_dict = convert(d['sources'])[src]
            sed_file = f.rstrip(".npy") + "_" + src.lower().replace(' ','_') + "_sed.npy"

            if path.exists(sed_file):
                sed = np.load(sed_file, allow_pickle=True, encoding='latin1').flat[0]
            else:
                sed = None

            logging.info(" ===== {0:s} = {1:s} ===== ".format(src, src_dict['assoc']['ASSOC1']))
            if select_src is not None:
                if not src == select_src:
                    continue

            if src_dict['ts'] < ts_thr:
                logging.warning("Source TS = {0:.2f} < thr."
                                " No templates will be generated".format(src_dict['ts']))
                continue
            c = SkyCoord(ra=src_dict['ra'], dec=src_dict['dec'], unit='deg', frame='icrs')

            log_energy_edges_eV = d['roi']['log_energies'] + 6.

            # prepend and append energy bins 
            d_log_e = np.diff(log_energy_edges_eV)[0]
            log_e_low = log_energy_edges_eV[0] - np.arange(1, n_ebins_add + 1, 1)[::-1] * d_log_e
            log_e_high = log_energy_edges_eV[-1] + np.arange(1, n_ebins_add + 1, 1) * d_log_e
            energy_edges_eV = 10.**np.concatenate([log_e_low, log_energy_edges_eV, log_e_high])

            width = d['config']['binning']['roiwidth']
            binsz = d['config']['binning']['binsz']

            # get source redshift either from best fit including EBL
            # or from a user-provided list
            if redshifts is None:
                try:
                    z = src_dict['spectral_pars']['redshift']['value']
                except KeyError:
                    logging.warning("redshift not in spectral pars dict and no redshift information given!")
                    raise
            else:
                z = redshifts[ifile]

            # get the index of the file corresponding to this redshift
            if use_closest_z:
                iz = np.argmin(np.abs(z - np.array(self._zList)))
                if np.abs(z - self._zList[iz]) > 0.005:
                    logging.info("Source redshift = {0:.3f}, nearest template redshift {1:.3f},"
                        " difference larger than 0.005, continuing".format(z, self._zList[iz]))
                    continue
            else:
                iz = self._zList.index(np.round(z, 3))

            if select_z is not None and not self._zList[iz] == select_z:
                continue

            logging.info("Source redshift = {0:.3f}, using template redshift {1:.3f}".format(z, self._zList[iz]))
            logging.info("Starting loop over B field parameters")
            for ib, b in enumerate(self._bList):
                if select_bfield is not None:
                    if not select_bfield == b:
                        continue
                for il, l in enumerate(self._turbScaleList):
                    for it, t in enumerate(self._th_jetList):
                        self.setOutput(idB=ib, idL=il, it=it, iz=iz)

                        if not path.exists(self.outputfile):
                            logging.warning("{0:s} not found. Simulations not finished yet?".format(self.outputfile))
                            continue

                        # generate new output file for chosen theta obs angle
                        fname = self.outputfile.replace('.hdf5', '_th_obs{0:.1f}.hdf5'.format(theta_obs))
                        if not path.isfile(fname) or overwrite:
                            data, config = stack_results_lso(infile=self.outputfile,
                                                             outfile=fname,
                                                             theta_obs=theta_obs,
                                                             use_cosmo=False,
                                                             )

                        # set up cascade map
                        if not dry:
                            casc = CascMap.gen_from_hd5f(fname, 
                                                         skycoord=c,
                                                         width=width, 
                                                         binsz=binsz,
                                                         ebins=energy_edges_eV * u.eV,
                                                         id_detection=22,
                                                         smooth_kwargs={'kernel': Gaussian2DKernel, 'threshold': 1, 'steps': 50}
                                                         )

                        # set the maximum delay time
                        logging.info("Applying time cut {0:.1e}".format(tmax))
                        if not dry:
                            casc.tmax = tmax * u.yr

                        # loop through spectral index and cut off energy, 
                        # set the weights, and export fits file
                        if "PowerLaw2" in src_dict['SpectrumType']:
                            scale = 1e9 * u.eV
                            prefactor = pl2_to_pl(src_dict, scale.to('MeV').value) * u.Unit("MeV-1 s-1 cm-2")
                        else:
                            prefactor = src_dict['spectral_pars']['Prefactor'] * u.Unit("MeV-1 s-1 cm-2")
                            scale = src_dict['spectral_pars']['Scale'] * u.MeV.to('eV')

                        pars = {'Prefactor': prefactor, 'Scale': scale}

                        logging.info("Starting loop over spectral parameters")

                        # apply a rotation
                        if not dry:
                            casc.rotation(Angle(rotation * u.deg))

                        #err = 0.1 * src_dict['spectral_pars']['Index']['value'] if np.isnan(src_dict['spectral_pars']['Index']['error']) \
                            #else src_dict['spectral_pars']['Index']['error']

                        for i, ecut in enumerate(Ecut_TeV):
                            for j, cs in enumerate(cov_scale):
                                #pars['Index'] = src_dict['spectral_pars']['Index']['value'] + cs * err
                                pars['Index'] = src_dict['spectral_pars']['Index']['value'] + cs
                                pars['Cutoff'] = (ecut * u.TeV).to('eV')

                                if np.any(np.isnan([v.value if isinstance(v, u.Quantity) else v for v in pars.values()])):
                                    raise ValueError("Parameters contain nans!")

                                suffix = "em_casc_{5:s}_tmax_{0:.0e}_theta_obs_{1:s}_rotation_{2:s}" \
                                         "_index_{3:s}_Ecut_{4:s}".format(tmax, str(theta_obs).replace('.', 'p'), 
                                                                          str(rotation).replace('.', 'p'),
                                                                          "{0:.3f}".format(pars["Index"]).replace('.', 'p'),
                                                                          str(ecut).replace('.', 'p'), 
                                                                          src.lower().replace(' ',''))

                                filename = path.join(path.dirname(self.outputfile), suffix + '.fits')

                                if path.exists(filename) and not overwrite:
                                    logging.info("{0:s} exists and overwrite is set to False. Continuing".format(filename))
                                    continue 

                                # set the weights
                                if not dry:
                                    casc.apply_spectral_weights(lambda E: inj_spec(E, **pars), smooth=True)

                                # plot the skymap and spectrum
                                # for one set of assumed spectral parameters
                                if make_plots and j == len(cov_scale) / 2 and i == len(Ecut_TeV) - 1:
                                #if make_plots:
                                    # skymap
                                    fig, ax, cax = casc.casc_obs.sum_over_axes(['energy']).plot(
                                        add_cbar=True, stretch='log', cmap='cubehelix_r')
                                    ax.tick_params(direction='out')
                                    fig.savefig(path.join(path.dirname(self.outputfile), suffix + '_skymap.png'), dpi=150)
                                    plt.close("all")

                                    # spectrum
                                    fig, ax = casc.plot_spectrum(energy_unit='MeV', E2dNdE_unit='MeV cm-2 s-1')

                                    cen = casc.casc.geom.get_axis_by_name('energy_true').center
                                    ax.loglog(cen.to('MeV'),
                                              (inj_spec(cen, **pars).to(casc.casc_obs.quantity.unit * u.sr) * cen ** 2. / (1. + z)).to('MeV cm-2 s-1'),
                                              label=r'injected $\gamma$-ray spectrum'
                                              )
                                    vy = ax.get_ylim()
                                    vx = ax.get_xlim()
                                    if sed is not None:
                                        SEDPlotter.plot_sed(sed, ax=ax)
                                        vy2 = ax.get_ylim()
                                        vx2 = ax.get_xlim()

                                    ax.set_ylim(vy[1] / 1e4, np.max([vy[1], vy2[1]]))
                                    ax.set_xlim(vx[0], vx[1])
                                    ax.legend(loc=1, fontsize='xx-small')
                                    ax.grid()
                                    fig.savefig(path.join(path.dirname(self.outputfile), suffix + '_spec.png'), dpi=150)
                                    plt.close("all")

                                # export to fits file
                                logging.info("writing fits template to {0:s}".format(filename))

                                extra_dict = {"spectral_parameters" :{k: v if isinstance(v, float) else v.to_string() for k, v in pars.items()},
                                              "sim_config": casc.config if not dry else {}
                                               }
                                if not dry:
                                    casc.export_casc_obs_to_fits(filename, extra_header_dict=extra_dict)
                        if not dry:
                            del casc
    def plot_templates(self,
                       fermipy_files,
                       select_z=None,
                       tmax=1e7,
                       theta_obs=0.,
                       select_b_field=None,
                       cov=2.,
                       index_step=0.5,
                       Ecut_TeV=np.arange(1.,17., 2.),
                       rotation=0.,
                       ts_thr=25., 
                       redshifts=None,
                       use_closest_z=True,
                       n_ebins_add=3,
                       cmap="cividis",
                       select_src=None,
                       overwrite=False):
        """
        Plot the IGMF templates

        Parameters
        ----------
        fermipy_files: list
            list of npy files that are the result of an ROI fermipy fit

        {options}
        select_z: float or None
            if not none, only generate templates for this redshift

        theta_obs: float
            Angle between jet axis and line of sight in degrees

        tmax: float
            Maximum allowed delay time of cascade photons in years.

        cov: float
            determines the range of the spectral index loop
            through index + cov * error

        index_step: float
            increment for spectral index loop,
            cov_loop = np.arange(-cov, cov + index_step, index_step)

        Ecut_TeV: array-like
            assumed exponential cutoff energies in TeV

        rotation: float
            Angle in degrees by which cascade template is rotated

        ts_thr: float
            Only consider sources for which fermipy analysis gave ts value larger than this 
            value

        use_closest_z: bool
            if True, use template with redshift closest to source redshift, 
            otherwise, redshift has to match exactly

        redshifts: array-like
            list with source redshifts. If not given, it's extracted from the fermipy
            files. This only works if the sources were fitted with EBL absorption

        dry: bool
            only perform template generation if this is False.
            Useful for debugging.

        overwrite: bool
            Overwrite existing templates.

        n_ebins_add: int
            number of energy bins that will be added at low and high energy end
        """
        from myplot.spectrum import SEDPlotter
        import matplotlib.pyplot as plt

        # the assumed intrinsic spectrum:
        # power law with exponential cut off
        inj_spec = lambda E, **p : p['Prefactor'] * (E / p['Scale']) ** (-p['Index']) * \
                                   np.exp(-E / p['Cutoff'])

        # the steps for the index: Index + (Index_err) * cov_scale
        cov_scale = np.arange(-cov, cov + index_step, index_step)

        cp = plt.cm.get_cmap(cmap)
        for ifile, f in enumerate(fermipy_files):
            if not path.exists(f):
                logging.warning("{0:s} not found".format(f))
                continue

            d = np.load(f, allow_pickle=True, encoding="latin1").flat[0]
            src = d['config']['selection']['target']
            src_dict = convert(d['sources'])[src]
            sed_file = f.rstrip(".npy") + "_" + src.lower().replace(' ','_') + "_sed.npy"

            if path.exists(sed_file):
                sed = np.load(sed_file, allow_pickle=True, encoding='latin1').flat[0]
            else:
                sed = None

            assoc = src_dict['assoc']['ASSOC1']
            logging.info(" ===== {0:s} = {1:s} ===== ".format(src, assoc))
            if select_src is not None:
                if not src == select_src:
                    continue

            if src_dict['ts'] < ts_thr:
                logging.warning("Source TS = {0:.2f} < thr."
                                " No templates will be generated".format(src_dict['ts']))
                continue
            c = SkyCoord(ra=src_dict['ra'], dec=src_dict['dec'], unit='deg', frame='icrs')

            log_energy_edges_eV = d['roi']['log_energies'] + 6.

            # prepend and append energy bins 
            d_log_e = np.diff(log_energy_edges_eV)[0]
            log_e_low = log_energy_edges_eV[0] - np.arange(1, n_ebins_add + 1, 1)[::-1] * d_log_e
            log_e_high = log_energy_edges_eV[-1] + np.arange(1, n_ebins_add + 1, 1) * d_log_e
            energy_edges_eV = 10.**np.concatenate([log_e_low, log_energy_edges_eV, log_e_high])

            width = d['config']['binning']['roiwidth']
            binsz = d['config']['binning']['binsz']

            # get source redshift either from best fit including EBL
            # or from a user-provided list
            if redshifts is None:
                try:
                    z = src_dict['spectral_pars']['redshift']['value']
                except KeyError:
                    logging.warning("redshift not in spectral pars dict and no redshift information given!")
                    raise
            else:
                z = redshifts[ifile]

            # get the index of the file corresponding to this redshift
            if use_closest_z:
                iz = np.argmin(np.abs(z - np.array(self._zList)))
                if np.abs(z - self._zList[iz]) > 0.005:
                    logging.info("Source redshift = {0:.3f}, nearest template redshift {1:.3f},"
                        " difference larger than 0.005, continuing".format(z, self._zList[iz]))
                    continue
            else:
                iz = self._zList.index(np.round(z, 3))

            if select_z is not None and not self._zList[iz] == select_z:
                continue

            logging.info("Source redshift = {0:.3f}, using template redshift {1:.3f}".format(z, self._zList[iz]))
            logging.info("Starting loop over B field parameters")
            iplot = 0
            nplots = len(cov_scale) + len(Ecut_TeV) 
            if select_b_field is None:
                nplots += len(self._bList)

            for ib, b in enumerate(self._bList):

                if select_b_field is not None and not b == select_b_field:
                    continue

                for il, l in enumerate(self._turbScaleList):
                    for it, t in enumerate(self._th_jetList):
                        self.setOutput(idB=ib, idL=il, it=it, iz=iz)

                        if not path.exists(self.outputfile):
                            logging.warning("{0:s} not found. Simulations not finished yet?".format(self.outputfile))
                            continue

                        # generate new output file for chosen theta obs angle
                        fname = self.outputfile.replace('.hdf5', '_th_obs{0:.1f}.hdf5'.format(theta_obs))
                        if not path.isfile(fname) or overwrite:
                            data, config = stack_results_lso(infile=self.outputfile,
                                                             outfile=fname,
                                                             theta_obs=theta_obs,
                                                             use_cosmo=False,
                                                             )

                        # set up cascade map
                        casc = CascMap.gen_from_hd5f(fname, 
                                                     skycoord=c,
                                                     width=width, 
                                                     binsz=binsz,
                                                     ebins=energy_edges_eV * u.eV,
                                                     id_detection=22,
                                                     smooth_kwargs={'kernel': Gaussian2DKernel, 'threshold': 4, 'steps': 50}
                                                     )

                        # set the maximum delay time
                        logging.info("Applying time cut {0:.1e}".format(tmax))
                        casc.tmax = tmax * u.yr

                        # loop through spectral index and cut off energy, 
                        # set the weights, and export fits file
                        if "PowerLaw2" in src_dict['SpectrumType']:
                            scale = 1e9 * u.eV
                            prefactor = pl2_to_pl(src_dict, scale.to('MeV').value) * u.Unit("MeV-1 s-1 cm-2")
                        else:
                            prefactor = src_dict['spectral_pars']['Prefactor'] * u.Unit("MeV-1 s-1 cm-2")
                            scale = src_dict['spectral_pars']['Scale'] * u.MeV.to('eV')

                        pars = {'Prefactor': prefactor, 'Scale': scale}

                        logging.info("Starting loop over spectral parameters")
                        logging.info("Ecut_TeV: {0}".format(Ecut_TeV))
                        logging.info("cov_scale: {0}".format(cov_scale))

                        # apply a rotation
                        casc.rotation(Angle(rotation * u.deg))

                        err = 0.1 * src_dict['spectral_pars']['Index']['value'] if np.isnan(src_dict['spectral_pars']['Index']['error']) \
                            else src_dict['spectral_pars']['Index']['error']

                        for i, ecut in enumerate(Ecut_TeV):
                            for j, cs in enumerate(cov_scale):
                                pars['Index'] = src_dict['spectral_pars']['Index']['value'] + cs * err
                                pars['Cutoff'] = (ecut * u.TeV).to('eV')

                                suffix = "em_casc_{5:s}_tmax_{0:.0e}_theta_obs_{1:s}_rotation_{2:s}" \
                                         "_index_{3:s}_Ecut_{4:s}".format(tmax, str(theta_obs).replace('.', 'p'), 
                                                                          str(rotation).replace('.', 'p'),
                                                                          "{0:.3f}".format(pars["Index"]).replace('.', 'p'),
                                                                          str(ecut).replace('.', 'p'), 
                                                                          src.lower().replace(' ',''))
                                # set the weights
                                casc.apply_spectral_weights(lambda E: inj_spec(E, **pars), smooth=True)

                                # skymap, only plot once
                                if not iplot:
                                    fig_sky, ax_sky, cax = casc.casc_obs.sum_over_axes(['energy']).plot(
                                        add_cbar=True, stretch='log', cmap=cmap)
                                    ax_sky.tick_params(direction='out')
                                    title = r"{0:s}, $t_\mathrm{{max}}$ = {1:.1e}, $\theta_\mathrm{{obs}} = {2:.1f}^\circ$, $\phi = {3:.1f}^\circ$".format(
                                                assoc, tmax, theta_obs, rotation)
                                    fig_sky.suptitle(title)
                                    ax_sky.grid(color="0.7", ls=":")
                                    fig_sky.savefig(path.join(path.dirname(self.outputfile), suffix + '_skymap.png'), dpi=150)

                                label = "$\Gamma = {0:.2f}, E_\mathrm{{cut}} = {1:.2f}$ TeV".format(pars["Index"], ecut)
                                label_casc = "$B = {0:.2f}$".format(b)

# TODO: pre calculate number of lines to use full color scale
# TODO: customize lables
# TODO use steps
# TODO check transparency so that observed spectrum is still visiblie
# TODO include IACT data points
                                # spectrum
                                col = cp(iplot / float(nplots))
                                ds = "steps-pre"
                                lw = 1.5
                                zorder=-2
                                if not iplot:
                                    fig_spec, ax_spec = casc.plot_spectrum(energy_unit='MeV',
                                                                           E2dNdE_unit='MeV cm-2 s-1',
                                                                           kwargs_casc=dict(label=label_casc, color=col, drawstyle=ds, lw=lw, marker='', ls='-', zorder=zorder),
                                                                           kwargs_prim=dict(plot=True, label='', color=col, lw=lw, marker='', ls='-', zorder=zorder),
                                                                           kwargs_tot=dict(plot=False, label='', color=col, drawstyle=ds, lw=lw),
                                                                           )
                                else:
                                    casc.plot_spectrum(energy_unit='MeV',
                                                       E2dNdE_unit='MeV cm-2 s-1',
                                                       ax=ax_spec,
                                                       fig=fig_spec,
                                                       kwargs_casc=dict(label=label_casc, color=col, drawstyle=ds, lw=lw, marker='', ls='-', zorder=zorder),
                                                       kwargs_prim=dict(plot=True, label='', color=col, lw=lw, marker='', ls='-', zorder=zorder),
                                                       kwargs_tot=dict(plot=False, label='', color=col, drawstyle=ds, lw=lw),
                                                       )

                                cen = casc.casc.geom.get_axis_by_name('energy_true').center

                                if (len(cov_scale) > 1 or len(Ecut_TeV) > 1) or iplot == 0:
                                    ax_spec.loglog(cen.to('MeV'),
                                                  (inj_spec(cen, **pars).to(casc.casc_obs.quantity.unit * u.sr) * cen ** 2. / (1. + z)).to('MeV cm-2 s-1'),
                                                  label=label,
                                                  color=col,
                                                  lw=lw
                                                  )

                                vy = ax_spec.get_ylim()
                                vx = ax_spec.get_xlim()
                                iplot += 1

            if sed is not None:
                SEDPlotter.plot_sed(sed, ax=ax_spec)
                vy2 = ax_spec.get_ylim()
                vx2 = ax_spec.get_xlim()

                ax_spec.set_ylim(vy[1] / 1e4, np.max([vy[1], vy2[1]]))
                ax_spec.set_xlim(vx[0], vx[1])

            ax_spec.legend(loc=1, fontsize='xx-small')
            ax_spec.grid()
            fig_spec.savefig(path.join(path.dirname(self.outputfile), suffix + '_spec.png'), dpi=150)

def main(**kwargs):
    usage = "usage: %(prog)s"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('-f', '--files', required=True, nargs="+")
    parser.add_argument('--tmax', default=1e7, help='max AGN duty cycle in years', type=float)
    parser.add_argument('--theta-obs', default=0., help='Angle between AGN jet axis and line of sight', type=float)
    args = parser.parse_args()

    utils.init_logging('INFO', color=True)
    with open(args.conf) as f:
        config = yaml.safe_load(f)
    generator = GenerateFitsTemplates(**config)
    return generator, args

if __name__ == '__main__':
    gen, args = main()
