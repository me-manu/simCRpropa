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
from gammapy.estimators import FluxPointsEstimator, FluxPoints
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
from scipy.interpolate import UnivariateSpline
from scipy.special import gammaincinv


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

def plot_likelihood_vs_B(config, select_source=None, skip_sources=[], simple_title=False, cmap=plt.cm.Set2, conf_level=0.95, dof=1):
    # loop through the sources:

    outdir = os.path.join(config['global']['outdir'], "tmax{0:.1e}".format(config['global']['tmax']))
    results = {}
    src_names = [] 

    # load the fermi combined cube in order to get the likelihood without halo 
    # as a proxy for a high B field
    llh_fermi = np.load(config['llh_fermi_file'], allow_pickle=True, encoding="latin1").flat[0])

    for src in config.keys():
        if src == 'global':
            continue

        if select_source is not None:
            if not src == args.select_source:
                continue

        if src in skip_sources:
            logging.info(f"Skipping source {src:s}")
            continue

        logging.info(f" ====== {src} ======= ")

        logging.info(f"Loading result for source {src:s}")
        try:
            results[src] = np.load(os.path.join(outdir, f"{src:s}.npz"), allow_pickle=True)
            src_names.append(src)

        except IOError:
            logging.error("File not found: {0:s}",format(os.path.join(outdir, f"{src:s}.npz")))
            continue

            #np.savez(os.path.join(outdir, f"{src:s}.npz"),
            #         combined=stat_results[src],
            #         fermi_only=stat_results_fermi_only[src],
            #         ps=stat_results_ps[src])
    tot_stat = np.array([results[src]['combined'] for src in src_names])
    b_fields = config['global']['b_fields']

    # likelihoods for each source
    b_vals = np.linspace(np.log10(np.min(b_fields)), np.log10(np.max(b_fields)), 50)
    tot_stat_interp = np.zeros_like(b_vals)

    # loop over sources
    for i, ts in enumerate(tot_stat):
        interp = UnivariateSpline(np.log10(b_fields), ts, k=2, s=0)
        tot_stat_interp += interp(b_vals)

    #    plt.semilogx(b_fields, ts - ts.min(), label=src_names[i], ls='--')
        plt.semilogx(10.**b_vals, interp(b_vals) - interp(b_vals).min(), label=src_names[i], color=cmap(i / float(len(b_fields))))
    
    summed_ts = np.sum(tot_stat, axis=0) - np.sum(tot_stat, axis=0).min()
    #plt.semilogx(b_fields, summed_ts, label="Summed", lw=2, color="k", ls='--')
    delta_ts = tot_stat_interp - tot_stat_interp.min()
    plt.semilogx(10.**b_vals, delta_ts, label="Summed", lw=2, color="k")

    # find the limit
    idx = np.where(delta_ts == 0)[0][0]  # interpolation up to minimum
    thrlim = 2. * gammaincinv(0.5 * dof, 1. - 2. * (1. - conf_level))
    roots = UnivariateSpline(b_vals[:idx], delta_ts[:idx] - thrlim, k=3, s=0)
    logging.info("{0:.2f}% confidence level = Delta TS = {1:.2f} lower limit on B: {2} G".format(conf_level, thrlim, np.power(10., roots.roots())))
    plt.axhline(thrlim, ls='--', color="0.5", lw=2)

    t_max_power = np.floor(np.log10(config['global']['tmax']))
    t_max_base = config['global']['tmax'] / 10.**t_max_power
    if simple_title:
        title = r"$t_\mathrm{{max}} = 10^{{{0:.0f}}}$ yrs".format(
                    t_max_power)
    else:
        title = r"$t_\mathrm{{max}}$ = {0:.2f}$\times10^{{{1:.0f}}}$ yrs, $\theta_\mathrm{{obs}} = {2:.1f}^\circ$, $\phi = {3:.1f}^\circ$".format(
                    t_max_base, t_max_power, 0., config['global']['rotation'])

    ax = plt.gca()
    ax.set_ylim(-.5, 15.)
    ax.grid(which="both", color="0.9")
    ax.legend(title=title, fontsize='x-small', ncol=3)
    ax.set_xlabel("$B$-field strength (G)")
    ax.set_ylabel("$-2\Delta\ln\mathcal{L}$")
    plt.tight_layout()

    return ax


def perform_casc_fit(config, geom, ps_model, llh, dataset,
                     on_radius=Angle("0.07 deg"), B=1e-16, initial_bias=0., plot=False):
    """
    Perform the combined cascade fit

    Parameters
    ----------
    config: dict
        Configuration dictionary

    B: float
        B-field strength

    geom:`~gammapy.maps.WcsGeom` object
        geometry of the dataset

    on_radius: `~astropy.coordinates.Angle` object
        Radius of ON region of IACT analysis

    ps_model: `gammapy.model.modeling.models.spectral` object
        Assumed point source spectral model for the intrinsic spectrum

    llh: `simCRpropa.fermiinterp.LogLikeCubeFermi`
        Interpolated Fermi likelihood cube

    dataset: `~gammapy.datasets.SpectrumOnOffDataset`
        The IACT data set

    initial_bias: float
        initial value for the energy bias (default: 0)

    plot: bool
        If true, generate diagnostic plots

    Returns
    -------
    Tuple containing -2 * ln (likelihood values) for fit
    on Fermi-LAT data only and combined fit
    """

    logging.info(f" ------ B = {B:.2e} ------- ")
    # Load the cascade
    # generate a casc map with low
    # spatial resolution to speed up calculations
    ebl = EBLAbsorptionNormSpectralModel.read(filename=config['global']['gammapy_ebl_file'],
                                              redshift=config[src]['z_src'])

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
                                     #ps_model.spectral_model.model1.copy(),
                                     ps_model.copy(),
                                     ebl, on_region,
                                     rotation=config['global']['rotation'] * u.deg,
                                     tmax=config['global']['tmax'] * u.yr,
                                     bias=0. * u.dimensionless_unscaled,
                                     energy_unit_spectral_model="TeV",
                                     use_gammapy_interp=False
                                     )
    # Plot the total model
    if plot:
        energy_power = 2
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)

        e_range = [1e-3, 10.] * u.TeV
        casc_spec.add_primary = True
        casc_spec.plot(ax=ax, energy_range=e_range, energy_power=energy_power)
        ps_model_combined = ps_model * ebl
        ps_model_combined.plot(ax=ax, energy_range=e_range, energy_power=energy_power)
        casc_spec.add_primary = False
        casc_spec.plot(ax=ax, energy_range=e_range, energy_power=energy_power)
        casc_spec.add_primary = True
        plt.ylim(1e-15, 3e-12)

        ax.grid()
        fig.savefig(f"plots/{src:s}_B{B}_casc+ps_models.png")
        plt.close("all")

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
    casc_model.parameters['bias'].value = initial_bias
    casc_model.parameters['bias'].frozen = config['global']['fix_bias']
    logging.info(f"Initial parameters:\n{casc_model.parameters.to_table()}")

    # interpolate the fermi likelihood for the right B field
    # TODO: changes necessary if more than one coherence length or theta jet used
    idb = np.where(llh.params["B"] == B)[0][0]
    llh.interp_llh(llh.params["B"][idb],  # choose a B field - should match casc file
                   llh.params["maxTurbScale"][0],  # choose a turb scale - should match casc file
                   llh.params["th_jet"][0],  # choose a jet opening angle - should match casc file
                   method='linear',
                   )
    # plot the interpolation
    if plot:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)

        # plot a likelihood surface
        # choose a slice in cut off energy
        cut_id = 0
        # build a grid of indices and norms
        ii, nn = np.meshgrid(llh._params["Index"], llh.log_norm_array, indexing='ij')
        # plot the log likehood grid
        dlogl = 2. * (llh._llh_grid[cut_id] - llh._llh_grid[cut_id].max())

        vmin = -100.
        # check if most points have higher dlogl,
        # and if so, adjust color scaling
        if dlogl[dlogl > vmin].size < 10:
            vmin = 3. * dlogl[dlogl < 0].max()

        im = ax.pcolormesh(ii, nn, dlogl, cmap=cmap_name, vmin=vmin, vmax=0)
        ax.annotate("$E_\mathrm{{cut}} = {0:.0f}$TeV".format(llh.params["Cutoff"][cut_id]),
                    xy=(0.05, 0.95), xycoords='axes fraction', color='w', va='top',
                    fontsize='x-large'
                    )
        plt.colorbar(im, label='$\ln\mathcal{L}$')
        ax.tick_params(direction='out')
        plt.xlabel("$\Gamma$")
        plt.ylabel("$\log_{10}(N)$")
        plt.grid(color='0.7', ls=':')
        plt.subplots_adjust(bottom=0.2, left=0.2)
        fig.savefig("plots/{1:s}_lnl_fermi_grid_Ecut{0:.0f}TeV_B{2}.png".format(
            llh.params["Cutoff"][cut_id], src, B))
        plt.close("all")
    # initialize prior data set
    logging.info("Initializing data set with priors")
    prior_stack = PriorSpectrumDatasetOnOff.from_spectrum_dataset_fermi_interp(dataset,
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
    logging.info(f"Total stat after fit {fit_result_casc.total_stat}")
    logging.info(f"Fermi logl after fit {prior_stack.llh_fermi}")

    # plot the flux points
    if plot:
        fig = plt.figure(dpi=150)
        ax = flux_points.plot(energy_power=2., label='data', marker='o')
        if not casc_model.parameters['bias'].value == 0.:
            fp = flux_points.table.copy()
            fp['e_ref'] *= 1. + casc_model.parameters['bias'].value
            fp['e_min'] *= 1. + casc_model.parameters['bias'].value
            fp['e_max'] *= 1. + casc_model.parameters['bias'].value
            fp_rescale = FluxPoints(fp)
            fp_rescale.plot(ax = ax, energy_power=2., label='data rescaled', marker='o', color='green')


        # plot the final model
        e_range = [1e-3, 10.] * u.TeV
        # total model
        casc_model.spectral_model.add_primary = True
        casc_model.spectral_model.plot(ax=ax, energy_range=e_range, energy_power=2, label='Total', color='k')
        casc_model.spectral_model.plot_error(ax=ax, energy_range=e_range, energy_power=2)
        # point source
        obs_model = casc_model.spectral_model.intrinsic_spectral_model * casc_model.spectral_model.ebl


        # cascade
        casc_model.spectral_model.add_primary = False
        casc_model.spectral_model.plot(ax=ax,
                                       energy_range=e_range,
                                       energy_power=2,
                                       label='Cascade',
                                       color='k',
                                       ls='-.')
        casc_model.spectral_model.add_primary = True

        if casc_model.parameters['bias'].value == 0.:
            obs_model.plot(ax=ax, energy_range=e_range, energy_power=2, label='Point source', color='k', ls='--')
        else:
            bias_value = casc_model.copy().parameters['bias'].value
            casc_model.parameters['bias'].value = 0.
            obs_model.plot(ax=ax, energy_range=e_range, energy_power=2, label='Point source', color='green', ls='--')
            casc_model.spectral_model.plot(ax=ax, energy_range=e_range, energy_power=2, label='Total, bias = 0', color='green', ls=':')

            # cascade
            casc_model.spectral_model.add_primary = False
            casc_model.spectral_model.plot(ax=ax,
                                           energy_range=e_range,
                                           energy_power=2,
                                           label='Cascade',
                                           color='green',
                                           ls='-.')
            casc_model.spectral_model.add_primary = True
            # reset bias
            casc_model.parameters['bias'].value = bias_value

        # fermi SED
        SEDPlotter.plot_sed(sed, ax=ax, ms=6.,
                            marker='.',
                            color='C1',
                            mec='C1',
                            alpha=1.,
                            noline=False,
                            band_alpha=0.2,
                            line_alpha=0.5,
                            band_linestyle='-',
                            label="Fermi",
                            flux_unit='TeV cm-2 s-1',
                            energy_unit='TeV',
                            print_name=False,
                            # apply bias to fermi data, in fit it's the other way around
                            #bias=casc_model.parameters['bias'].value
                            )


        ax.legend(loc='lower center')
        plt.ylim(3e-15, 2e-10)
        plt.xlim(1e-3, 2e1)
        fig.savefig(f"plots/final_fits_{src:s}_b{B}.png")
        plt.close("all")

    # plot a likelihood surface for best cut value
    if plot:
        par_amp = prior_stack.models['casc'].parameters['amplitude']
        amp = np.logspace(np.log10(par_amp.value) - 1., np.log10(par_amp.value) + .5, 100)
        amp *= par_amp.unit

        par_idx = prior_stack.models['casc'].parameters['index']
        idx = np.linspace(par_idx.value - .5, par_idx.value + 1., 120)
        idx *= par_idx.unit
        ii, aa = np.meshgrid(idx, amp, indexing='ij')
        c_stat = prior_stack.get_llh_fermi(amplitude=aa.reshape(-1), index=ii.reshape(-1), set_attribute=False).reshape(aa.shape)
        #ii, nn = np.meshgrid(idx, llh.log_norm_array, indexing='ij')
        #c_stat = np.zeros((idx.shape[0], amp.shape[0]))
        #for i, iidx in enumerate(idx):
            #for j, a in enumerate(amp):
                #c_stat[i,j] = prior_stack.get_llh_fermi(amplitude=a, index=iidx, set_attribute=False)
        #c_stat = np.zeros((idx.shape[0], llh.log_norm_array.shape[0]))
        #for i, iidx in enumerate(idx.value):
        #    for j, log_norm in enumerate(llh.log_norm_array):
        #        c_stat[i,j] = prior_stack.get_llh_fermi(amplitude=10.**log_norm * u.Unit('MeV-1 s-1 cm-2'),
        #                                                index=iidx * u.dimensionless_unscaled,
        #                                                set_attribute=False,
        #                                                reference=prior_stack._ref_energy * u.MeV)
        d_cstat = 2. * (c_stat - c_stat.min())

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)

        vmin = 0.
        vmax = 300.

        im = ax.pcolormesh(ii.value, np.log10(aa.value), d_cstat, cmap=cmap_name, vmin=vmin, vmax=vmax)
        #im = ax.pcolormesh(ii, nn, d_cstat, cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.annotate("$E_\mathrm{{cut}} = {0:.2f}$TeV, $B={1:.2e}$G".format(1. / prior_stack.models['casc'].parameters['lambda_'].value, B),
                    xy=(0.05, 0.95), xycoords='axes fraction', color='k', va='top',
                    fontsize='x-large'
                    )
        plt.colorbar(im, label='$-2\Delta\ln\mathcal{L}$')
        ax.plot(par_idx.value, np.log10(par_amp.value), ms=10., marker='*', mec='k')
        ax.tick_params(direction='out')
        plt.xlabel("$\Gamma$")
        plt.ylabel("$\log_{10}(N_\mathrm{H.E.S.S.}) = \log_{10}((E_{0,{Fermi}} / E_{0,\mathrm{H.E.S.S.}})^{-\Gamma}N_{Fermi})$", fontsize='medium')
        plt.grid(color='0.7', ls=':')
        plt.subplots_adjust(bottom=0.2, left=0.2)
        fig.savefig("plots/{0:s}_lnl_fermi_interp_B{1}.png".format(src, B))
        plt.close("all")

    return prior_stack, prior_stack.llh_fermi, fit_result_casc.total_stat

if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the combined fermi and HESS fit"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('--select-source')
    parser.add_argument('--select-bfield', type=float)
    parser.add_argument('--plots', action="store_true", help="Create diagnostic plots")
    parser.add_argument('--overwrite', action="store_true", help="Overwrite result files")

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

    # initialize result arrays
    stat_results = {src: np.zeros(len(b_fields)) for src in config.keys() if not src == 'global'}
    stat_results_fermi_only = {src: np.zeros(len(b_fields)) for src in config.keys() if not src == 'global'}
    stat_results_ps = {src: np.zeros(len(b_fields)) for src in config.keys() if not src == 'global'}
    stat_results_tot = np.zeros(len(b_fields))

    # set up outdir 
    outdir = os.path.join(config['global']['outdir'], "tmax{0:.1e}".format(config['global']['tmax']))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # loop through the sources:
    for src in config.keys():
        if src == 'global':
            continue

        if args.select_source is not None:
            if not src == args.select_source:
                continue

        # delete casc model for next source
        # in order to start with correct initial parameters 
        try:
            del dataset
        except NameError:
            pass

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
        #epl.parameters['lambda_'].min = 0.
        #epl.parameters['lambda_'].max = 1.
        epl.parameters['lambda_'].min = 1. / llh.params['Cutoff'].max()
        epl.parameters['lambda_'].max = 1. / llh.params['Cutoff'].min()

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

        # save fit point source fit result
        stat_results_ps[src] = fit_result_ps.total_stat
        logging.info(f"Total stat of point source fit: {fit_result_ps.total_stat}")

        # compute flux points
        e_min, e_max = dataset_stack[0].energy_range[0].value, 15.
        energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 10) * u.TeV

        fpe = FluxPointsEstimator(energy_edges=energy_edges, source=ps_model.name)
        flux_points = fpe.run(datasets=dataset_stack)
        flux_points.table["is_ul"] = flux_points.table["ts"] < 4

        # plot the model and flux points
        if args.plots:
            fig = plt.figure(dpi=150)
            ax = fig.add_subplot(111)
            energy_power = 2
            flux_points.plot(ax=ax, energy_power=energy_power)
            ps_model.spectral_model.plot(ax=ax, energy_range=[e_min, e_max] * u.TeV, energy_power=energy_power)
            ps_model.spectral_model.plot_error(ax=ax, energy_range=[e_min, e_max] * u.TeV, energy_power=energy_power)
            ax.grid()
            fig.savefig(f"plots/{src:s}_ebl_epl.png")
            plt.close("all")

        # loop over magnetic fields
        if not os.path.exists(os.path.join(outdir, f"{src:s}.npz")) or args.overwrite:
            for ib, B in enumerate(b_fields[::-1]):  # start with largest field, closest to point source case

                try:  # re use best-fit parameters from fit before
                    intrinsic_model = dataset.models[0].spectral_model.intrinsic_spectral_model.copy()
                    initial_bias = dataset.models[0].spectral_model.parameters['bias'].value
                except NameError:  # start with best fit parameters from point source fit from IACT
                    intrinsic_model = ps_model.spectral_model.model1.copy()  # intrinsic model from TeV fit
                    initial_bias = 0.

                if args.select_bfield is not None:
                    if not B == args.select_bfield:
                        continue

                ds = dataset_stack[0].copy()

                dataset, llh_fermi, llh_combined = perform_casc_fit(config,
                                                                    geom,
                                                                    intrinsic_model,
                                                                    llh,
                                                                    ds,
                                                                    on_radius=on_radius,
                                                                    B=B,
                                                                    initial_bias=initial_bias,
                                                                    plot=args.plots)

                # save total stat results
                stat_results[src][len(b_fields) - 1 - ib] = llh_combined
                stat_results_tot[len(b_fields) - 1 - ib] += llh_combined
                stat_results_fermi_only[src][len(b_fields) - 1 - ib] = llh_fermi

                # save best fit parameters
                pars_out_file = os.path.join(outdir, f"best_fit_pars_{src:s}_B{B:.2e}_fix_bias{config['global']['fix_bias']}.fits")
                dataset.models.parameters.to_table().write(pars_out_file, overwrite=True)

            if args.select_bfield is None:  # only save if all B fields are computed 
                logging.info(f"Saving result for source {src:s}")
                np.savez(os.path.join(outdir, f"{src:s}.npz"),
                         combined=stat_results[src],
                         fermi_only=stat_results_fermi_only[src],
                         ps=stat_results_ps[src])
        else:
            res = np.load(os.path.join(outdir, f"{src:s}.npz"), allow_pickle=True)
            stat_results_tot += res['combined']

    if not os.path.exists(os.path.join(outdir, f"tot.npy")) or args.overwrite:
        if args.plots:
            plot_likelihood_vs_B(config)  # plot the likelihood profile
            plt.savefig("plots/lnl_vs_B_tmax{tmax:.0e}_fix-bias{fix_bias:b}.png".format(**config['global']),
                        dpi=150)
            plt.close("all")
        if args.select_bfield is None and args.select_source is None:  # only save if all B fields and sources are computed
            logging.info("Saving total result")
            np.save(os.path.join(outdir, "tot.npy"), stat_results_tot)

# TODO
#  - make sure that bias implementation is correct
