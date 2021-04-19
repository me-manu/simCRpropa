import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob
import argparse
import os
from gammapy.datasets import Datasets
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel, 
    SkyModel,
    EBLAbsorptionNormSpectralModel
)
from gammapy.modeling import Fit
from gammapy.estimators import FluxPointsEstimator
from astropy.convolution import Gaussian2DKernel
from astropy import units as u
from simCRpropa.cascmaps import CascMap
from simCRpropa.cascmodels import (
    CascadeSpectralModel,
    PriorSpectrumDatasetOnOff
)
from regions import CircleSkyRegion
from astropy.coordinates import Angle
from simCRpropa.fermiinterp import LogLikeCubeFermi
from myplot.spectrum import SEDPlotter
from matplotlib.colors import LinearSegmentedColormap


def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data


# set logging level
def init_logging(level="INFO"):
    logging.basicConfig(level=level,
                        stream=sys.stderr,
                        format='\033[0;36m%(filename)10s:\033[0;35m%(lineno)4s\033[0;0m --- %(levelname)7s: %(message)s')
    logging.addLevelName(logging.DEBUG, 
                         "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.addLevelName(logging.INFO, 
                         "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.WARNING, 
                         "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, 
                         "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

# Create my own colormap
colors = [
            (0, 0, 0),
            plt.cm.tab10(0.),
            plt.cm.tab10(0.1),
            (1, 1, 1)
         ] 
cmap_name = "my_cmap"
cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=200)
plt.register_cmap(cmap=cm)


if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the combined fermi and HESS fit"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('--select-source')
    parser.add_argument('--select-bfield', type=float)

    args = parser.parse_args()
    init_logging()

    with open(args.conf) as f:
        config = yaml.safe_load(f)

    # the region in which
    # we'll calculate the cascade contribution
    on_radius = Angle("{0:f} deg".format(config['global']['on_radius']))

    # the magnetic fields
    b_fields = config['global']['b_fields']

    # suppress runtime warnings
    np.seterr(divide='ignore', invalid='ignore')

    # loop through the sources:
    for src in config.keys():
        if src == 'global':
            continue

        if args.select_source is not None:
            if not src == args.select_source:
                continue

        logging.info(f" ====== {src} ======= ")

        logging.info("Loading IACT datasets...")
        # read 3d data set
        dataset_3d = Datasets.read(config['global']['iact_dataset_3d'].replace("*", src))
        geom = dataset_3d[0].geoms['geom']
        on_region = CircleSkyRegion(center=geom.center_skydir, radius=on_radius)

        # Load the 1D data set
        dataset_1d_file = config['global']['iact_dataset_1d'].replace("*", src)
        if os.path.exists(dataset_1d_file.replace(".yaml", "_stacked.yaml")):
            logging.info("Loading stacked dataset...")
            dataset_stack = Datasets.read(dataset_1d_file.replace(".yaml", "_stacked.yaml"))
        else:
            dataset_1d = Datasets.read(config['global']['iact_dataset_1d'].replace("*", src))
            # stack reduce the data set
            logging.info("stacking datasets...")
            dataset_stack = Datasets([dataset_1d.stack_reduce()])
            dataset_stack.write(dataset_1d_file.replace(".yaml", "_stacked.yaml"))

        # load fermi SED for plotting
        logging.info("Loading Fermi files")
        sed_file = glob.glob(config['global']['fermi_sed'].replace("*", src, 1))[0]
        sed = np.load(sed_file, allow_pickle=True, encoding='latin1').flat[0]

        # load fermi best fit
        avg_file = config['global']['fermi_avg'].replace("*", src)
        d = np.load(avg_file,
                    allow_pickle=True, encoding="latin1").flat[0]
        src_fgl_name = d['config']['selection']['target']
        src_dict = convert(d['sources'])[src_fgl_name]

        # important to get the scale
        # since interpolation of fermi llh was done with prefactor
        # corresponding to this scale
        logging.info("Loading Fermi Likelihood cube")
        llh_file = config['global']['llh_fermi_file']
        llh = LogLikeCubeFermi(llh_file)
        llh.get_llh_one_source(src, norm_key='dnde_src')

        # Define the model for the intrinsic / observed blazar spectrum
        epl = ExpCutoffPowerLawSpectralModel(
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            index=1.6 * u.dimensionless_unscaled,
            lambda_=0.1 * u.Unit("TeV-1"),
            reference=1 * u.TeV,
        )
        epl.parameters['lambda_'].min = 0.
        epl.parameters['lambda_'].max = 1.

        ebl = EBLAbsorptionNormSpectralModel.read(filename=config['global']['gammapy_ebl_file'],
                                                  redshift=config[src]['z_src'])

        obs = epl * ebl

        # fit the point source model
        # without the cascade
        logging.info("Fitting IACT data with point source")
        ps_model = SkyModel(spectral_model=obs.copy(), name='ps')
        dataset_stack[0].models = ps_model
        fit_1d = Fit(dataset_stack)
        fit_result_ps = fit_1d.run(optimize_opts=dict(print_level=1,
                                                      tol=0.1,
                                                      migrad_opts=dict(ncall=1000)
                                                      )
                                   )

        ps_model.parameters.to_table()
        ps_model_table = ps_model.spectral_model.model1.parameters.to_table()
        logging.info(f"Final point source parameters:\n{ps_model_table}")

        # compute flux points
        e_min, e_max = dataset_stack[0].energy_range[0].value, 15.
        energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 10) * u.TeV

        fpe = FluxPointsEstimator(energy_edges=energy_edges, source=ps_model.name)
        flux_points = fpe.run(datasets=dataset_stack)
        flux_points.table_formatted
        flux_points.table["is_ul"] = flux_points.table["ts"] < 4

        # plot the model and flux points
        #flux_points.plot(energy_power=2.)
        #ps_model.spectral_model.plot(energy_range=[e_min, e_max] * u.TeV, energy_power=2)
        #ps_model.spectral_model.plot_error(energy_range=[e_min, e_max] * u.TeV, energy_power=2.)

        # loop over magnetic fields
        for ib, B in enumerate(b_fields):
            if args.select_bfield is not None:
                if not B == args.select_bfield:
                    continue

            logging.info(f" ------ B = {B:.2e} ------- ")

            # Load the cascade
            # generate a casc map with low
            # spatial resolution to speed up calcuations
            casc_file = config['global']['casc_file'].replace("*", "{0:.3f}".format(config[src]['z_casc']), 1)
            casc_file = casc_file.replace("*", "{0:.2e}".format(B))
            casc_1d = CascMap.gen_from_hd5f(casc_file,
                                            skycoord=geom.center_skydir,
                                            width=2 * np.round(on_radius.value / geom.pixel_scales[0].value, 0) *
                                                            geom.pixel_scales[0].value,
                                            binsz=geom.pixel_scales[0].value,
                                            ebins=21,
                                            id_detection=22,
                                            smooth_kwargs={'kernel': Gaussian2DKernel, 'threshold': 2., 'steps': 50}
                                            )

            # Initialize the cascade model
            # use the best fit model for the intrinsic spectrum
            logging.info("Initializing cascade model ...")
            casc_spec = CascadeSpectralModel(casc_1d,
                                             ps_model.spectral_model.model1.copy(),
                                             ebl, on_region,
                                             rotation=config['global']['rotation'] * u.deg,
                                             tmax=config['global']['tmax'] * u.yr,
                                             bias=0. * u.dimensionless_unscaled,
                                             energy_unit_spectral_model="TeV",
                                             use_gammapy_interp=False
                                             )


            # Plot the total model
            #e_range = [1e-3, 10.] * u.TeV
            #casc_spec.add_primary = True
            #casc_spec.plot(energy_range=e_range, energy_power=2)
            #ps_model.spectral_model.plot(energy_range=e_range, energy_power=2)
            #casc_spec.add_primary = False
            #casc_spec.plot(energy_range=e_range, energy_power=2)
            #casc_spec.add_primary = True
            #plt.ylim(1e-15, 3e-12)

            # Perform the fit with the cascade
            casc_model = SkyModel(spectral_model=casc_spec, name='casc')

            # limit the parameter ranges and change the tolerance

            # index
            casc_model.parameters['index'].min = llh.params['Index'].min()
            casc_model.parameters['index'].max = llh.params['Index'].max()

            # amplitude
            casc_model.parameters['amplitude'].min = casc_model.parameters['amplitude'].value / 10.
            casc_model.parameters['amplitude'].max = casc_model.parameters['amplitude'].value * 10.

            # cutoff
            casc_model.parameters['lambda_'].min = 1. / llh.params['Cutoff'].max()
            casc_model.parameters['lambda_'].max = 1. / llh.params['Cutoff'].min()
            casc_model.parameters['lambda_'].frozen = config['global']['fix_cutoff']

            # bias between Fermi and HESS
            casc_model.parameters['bias'].value = 0.
            casc_model.parameters['bias'].frozen = config['global']['fix_bias']

            logging.info(f"Initial parameters:\n{casc_model.parameters.to_table()}")

            # interpolate the fermi likelihood for the right B field
            # TODO: changes necessary if more than one coherence length or theta jet used
            idb = np.where(llh.params["B"] == B)[0][0]
            llh.interp_llh(llh.params["B"][idb],   # choose a B field - should match casc file
                           llh.params["maxTurbScale"][0],   # choose a turb scale - should match casc file
                           llh.params["th_jet"][0],  # choose a jet opening angle - should match casc file
                           method='linear',
                          )

            # plot the interpolation
            #plt.figure(dpi=150)
            ## plot a likelihood surface
            ## choose a slice in cut off energy
            #cut_id = 1
            ## build a grid of indices and norms
            #ii, nn = np.meshgrid(llh._params["Index"], llh.log_norm_array, indexing='ij')
            ## plot the log likehood grid
            #dlogl = 2. * (llh._llh_grid[cut_id] - llh._llh_grid[cut_id].max())
            #im = plt.pcolormesh(ii, nn, dlogl, cmap=cmap_name, vmin=-10, vmax=0)
            #plt.annotate("$E_\mathrm{{cut}} = {0:.0f}$TeV".format(llh.params["Cutoff"][cut_id]),
                         #xy=(0.05,0.95), xycoords='axes fraction', color='w', va='top',
                         #fontsize='x-large'
                         #)
            #plt.colorbar(im, label='$\ln\mathcal{L}$')
            #plt.gca().tick_params(direction='out')
            #plt.xlabel("$\Gamma$")
            #plt.ylabel("$\log_{10}(N)$")
            #plt.grid(color='0.7', ls=':')
            #plt.subplots_adjust(bottom=0.2, left=0.2)
            ##plt.savefig("lnl_fermi_grid_Ecut{0:.0f}TeV.png".format(llh.params["Cutoff"][cut_id]), dpi=150)
            #

            # initialize prior data set
            logging.info("Initializing data set with priors")
            ds = dataset_stack[0].copy()
            prior_stack = PriorSpectrumDatasetOnOff.from_spectrum_dataset_fermi_interp(ds,
                                                                                       llh_fermi_interp=None
                                                                                       )
            # add the interpolator to the data set
            prior_stack.llh_fermi_interp = llh.interp

            # add the reference energy at which interpolation was performed (in MeV)
            prior_stack.ref_energy = src_dict['spectral_pars']['Scale']['value']

            prior_stack.models = casc_model

            logging.info("Performing combined fit")
            fit_casc = Fit([prior_stack])
            fit_result_casc = fit_casc.run(optimize_opts=dict(print_level=2,
                                                              tol=10.,
                                                              migrad_opts=dict(ncall=1000)
                                                              )
                                           )
            logging.info(f"Parameters after fit:\n{casc_model.parameters.to_table()}")

            # plot the flux points
            #ax = flux_points.plot(energy_power=2., label='data', marker='o')

            # plot the final model
            #e_range = [1e-3, 10.] * u.TeV
            # total model
            #casc_model.spectral_model.add_primary = True
            #casc_model.spectral_model.plot(energy_range=e_range, energy_power=2, label='Total', color='k')
            #casc_model.spectral_model.plot_error(energy_range=e_range, energy_power=2)
            # point source
            #obs_model = casc_model.spectral_model.intrinsic_spectral_model * casc_model.spectral_model.ebl
            #obs_model.plot(energy_range=e_range, energy_power=2, label='Point source', color='k', ls='--')

            # cascade
            #casc_model.spectral_model.add_primary = False
            #casc_model.spectral_model.plot(energy_range=e_range, energy_power=2, label='Cascade', color='k', ls='-.')
            #casc_model.spectral_model.add_primary = True

            # fermi SED
            #SEDPlotter.plot_sed(sed, ax=ax, ms=6.,
                                #marker='o',
                                #color='0.7',
                                #mec='0.7',
                                #alpha=1.,
                                #noline=False,
                                #band_alpha=0.5,
                                #line_alpha=1.,
                                #band_linestyle='-',
                                #label="Fermi",
                                #flux_unit='TeV cm-2 s-1',
                                #energy_unit='TeV',
                                #print_name=False
                                #)

            #plt.legend(loc='lower center')
            #plt.ylim(3e-15, 2e-10)
            #plt.xlim(1e-3, 2e1)

# TODO
#  - Save results
#  - make sure that bias implementation is correct
#  - make control plots: likelihood surface, PS fit, combined fit with halo



