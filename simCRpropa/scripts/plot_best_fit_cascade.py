import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob
import argparse
import os
from gammapy.datasets import Datasets
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel, 
    SkyModel,
    EBLAbsorptionNormSpectralModel
)
from gammapy.modeling import Fit
from gammapy.estimators import FluxPointsEstimator, FluxPoints
from gammapy.modeling.models import Models
from gammapy.irf import EnergyDependentTablePSF, PSFMap
from astropy.convolution import Gaussian2DKernel
from astropy import units as u
from astropy.table import Table
from astropy.visualization import wcsaxes
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
from matplotlib.patheffects import withStroke
effect = dict(path_effects=[withStroke(foreground="w", linewidth=2)])
effect_k = dict(path_effects=[withStroke(foreground="k", linewidth=2)])


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

def get_base_power(x):
    power = np.floor(np.log10(x))
    base = x / 10.**power
    return base, power

def plot_obs_sed(dataset, sed_fermi, ax=None, fig=None, plot_fermi_fp=False):
    """
    Plot the observed sed

    Parameters
    ----------
    dataset: gammapy dataset
        Observed dataset with model

    sed_fermi: dict
        Dict with SED from Fermi

    Returns
    -------
    Axis object with SED
    """

    # compute flux points
    e_min, e_max = dataset.energy_range[0].value, 15.
    energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 10) * u.TeV
    model = dataset.models[0]

    fpe = FluxPointsEstimator(energy_edges=energy_edges, source=model.name)
    dataset.models[0].datasets_names = None
    flux_points = fpe.run(datasets=dataset)
    flux_points.table["is_ul"] = flux_points.table["ts"] < 4

    if fig is None:
        fig = plt.figure(figsize=(5.5,4), dpi=150)
    if ax is None:
        ax = fig.add_subplot(111)

    # plot flux points and IACT model
    energy_power = 2
    col_hess = plt.cm.Set3(0.25)
    flux_points.plot(ax=ax,
                     energy_power=energy_power,
                     flux_unit="TeV-1 cm-2 s-1",
                     color=plt.cm.Set1(0.),
                     marker='o',
                     label="H.E.S.S.",
                     ms=5.,
                     )
    model.spectral_model.plot(ax=ax,
                              energy_range=[e_min, e_max] * u.TeV,
                              energy_power=energy_power,
                              flux_unit="TeV-1 cm-2 s-1",
                              color=col_hess, zorder=-1
                              )
    model.spectral_model.plot_error(ax=ax,
                                    energy_range=[e_min, e_max] * u.TeV,
                                    energy_power=energy_power,
                                    flux_unit="TeV-1 cm-2 s-1",
                                    alpha=1., 
                                    color=col_hess,
                                    zorder=-1, 
                                    ls='-'
                                    )
    xlim, ylim = list(ax.get_xlim()), list(ax.get_ylim())

    # plot the Fermi SED
    col_fermi=plt.cm.Set3(0.35)
    emax = SEDPlotter.plot_flux_points(sed, ms=6., ax=ax,
                        marker='o',
                        color=plt.cm.Set1(0.125),
                        energy_unit="TeV",
                        flux_unit="TeV cm-2 s-1",
                        alpha=1.,
                        noline=False,
                        band_alpha=1.,
                        line_alpha=1.,
                        band_linestyle='-', 
                        label="LAT",
                        do_plot=plot_fermi_fp,
                        plot_all_uls=False,
                        plot_uls_below_last_fp=True,
                        print_name=False,
                        )
    SEDPlotter.plot_model(sed['model_flux'], ms=6., ax=ax,
                        marker='o',
                        color=col_fermi,
                        energy_unit="TeV",
                        flux_unit="TeV cm-2 s-1",
                        emax=emax,
                        alpha=1.,
                        noline=False,
                        band_alpha=1.,
                        line_alpha=1.,
                        band_linestyle='-', 
                        label="LAT" if not plot_fermi_fp else '',
                        zorder=-1
                        )
    xlim[0] = np.min([xlim, ax.get_xlim()])
    xlim[1] = np.max([xlim, ax.get_xlim()])
    ylim[0] = np.min([ylim, ax.get_ylim()])
    ylim[1] = np.max([ylim, ax.get_ylim()])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend(loc="best")

    energy_unit = 'TeV'
    ax.set_xlabel('Energy ({0:s})'.format(energy_unit))
    ax.set_ylabel('$E^{{2}} dN/dE$ ({0:s} cm$^{{-2}}$ s$^{{-1}}$)'.format(energy_unit))

    return fig, ax, flux_points

def plot_best_fit_cascade(config, geom, ps_model, dataset, parameters,
                          psf_file=None,
                          flux_points=None,
                          on_radius=Angle("0.07 deg"), B=1e-16, ax_sed=None):
    """
    Plot the best fit for the cascade

    Parameters
    ----------
    config: dict
        Configuration dictionary

    geom:`~gammapy.maps.WcsGeom` object
        geometry of the dataset

    on_radius: `~astropy.coordinates.Angle` object
        Radius of ON region of IACT analysis

    ps_model: `gammapy.model.modeling.models` object
        Assumed point source model

    dataset: `~gammapy.datasets.SpectrumOnOffDataset`
        The IACT data set


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
                                    width=1. * u.deg,
                                    binsz=geom.pixel_scales[0].value / 2.,
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

    # Set values to best fit
    casc_model = SkyModel(spectral_model=casc_spec, name='casc')
    # read in parameters from best fit
    for name in parameters.names:
        casc_spec.parameters[name].value = parameters[name].value
        casc_spec.parameters[name].error = parameters[name].error
        casc_spec.parameters[name].min = parameters[name].max
        casc_spec.parameters[name].max = parameters[name].min
        casc_spec.parameters[name].frozen = parameters[name].frozen
        casc_spec.parameters[name].unit = parameters[name].unit

    # evaluate model once to set weights
    kwargs = {name : parameters[name].quantity for name in parameters.names}
    _ = casc_spec.evaluate(dataset.geoms['geom'].axes['energy'].center, **kwargs)

    # do theta sq plot
    # get separations:
    sep = casc_spec.cascmap.casc_obs.geom.separation(casc_spec.cascmap.casc_obs.geom.center_skydir)
    theta_sq_edges_deg = np.arange(0., sep.value.max()**2., 0.005 / 2.)
    #theta_sq_edges_deg = np.arange(0., 0.2, 0.005 / 2.)
    ibin = np.digitize(sep.value**2., theta_sq_edges_deg) - 1
    flux = np.zeros(theta_sq_edges_deg.size - 1)

    m_energy = casc_spec.cascmap.casc_obs.geom.axes['energy_true'].edges.to(u.TeV) > dataset.energy_range[0]
    integral_flux_map = casc_spec.cascmap.casc_obs * casc_spec.cascmap.casc_obs_bin_volume
    for i in range(ibin.max()):
        m = ibin == i
        # integrate over energy, for energies larger than threshold
        integral_flux = integral_flux_map.data[m_energy[:-1]].sum(axis=0)

        # sum over spatial bins with correct separation
        flux[i] = np.sum(integral_flux[m])

    # plot flux vs theta_sq
    theta_sq_center_deg = 0.5 * (theta_sq_edges_deg[:-1] + theta_sq_edges_deg[1:])
    plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    ax.errorbar(theta_sq_center_deg, flux,
                ls="None", marker='.', 
                xerr=[theta_sq_center_deg - theta_sq_edges_deg[:-1], theta_sq_edges_deg[1:] - theta_sq_center_deg], 
                label="Cascade model"
                )
    ax.set_xlabel(r"$\theta^2$ (deg$^2$)")
    ax.set_ylabel("Flux ({0:s})".format(integral_flux_map.unit))
    ax.axvline(0.005, color='red', ls='--', label=r"$\theta^2 = 0.005$")
    ax.axvline(0.01, color='red', ls=':', label=r"$\theta^2 = 0.01$")
    ax.set_xlim(0., 0.1)
    ax.legend(loc='best')

    # plot cdf
    ax2 = ax.twinx()
    cdf = np.cumsum(flux)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    ax2.plot(theta_sq_center_deg, cdf, color="k", drawstyle='steps')
    ax2.set_ylabel("CDF")
    ax2.set_xlim(0., 0.1)
    ax2.grid()
    ax2.annotate(f"$\mathbf{{{src:s}}}$", xy=(0.95,0.95), va='top', color='w', xycoords="axes fraction", **effect)
    ax2.annotate(r"$B={0[0]:.1f} \times 10^{{{0[1]:0n}}}$G".format(get_base_power(B)),
                xy=(0.95,0.9), va='top', xycoords="axes fraction", color='w', fontsize='small', **effect)
    ax2.annotate(r"$t_\mathrm{{max}}=10^{{{0[1]:0n}}}$ yrs".format(get_base_power(config['global']['tmax'])),
                    xy=(0.95,0.85), va='top', color='w', xycoords="axes fraction", fontsize='small', **effect)

    plt.savefig(f"plots/theta_sq{src:s}_b{B}_t{config['global']['tmax']:.0e}_EgtEthr.png")
    plt.close("all")

    # Sky map plot

    if psf_file is not None:
        # get the fermi kernel
         psf_table = EnergyDependentTablePSF.read(psf_file)
         r_psf = psf_table.containment_radius(1. * u.GeV)
         psf = PSFMap.from_energy_dependent_table_psf(psf_table)
         geom = casc_spec.cascmap.casc_obs.geom
         psf_kernel = psf.get_psf_kernel(position=geom.center_skydir,
                                         geom=geom,
                                         max_radius=geom.width[0,0] / np.sqrt(2.)
                                         )
         # do convolution
         conv_map = casc_spec.cascmap.casc_obs.convolve(psf_kernel).sum_over_axes(['energy_true'])
         # don't plot it, too broad
         #fig, ax, cbar = conv_map.plot(cmap=cmap_name, stretch='log', add_cbar=True)
    # the map
    fig, ax, cbar = casc_spec.cascmap.casc_obs.reduce_over_axes().plot(cmap=cmap_name, stretch='log', add_cbar=True)

    ax.tick_params(direction="out")
    ax.grid(color="0.5", ls=":")
    ticks = np.array(cbar.get_ticks()[::2])
    cbar.set_ticks(ticks)
    power = np.floor(np.log10(ticks[1]))
    cbar.set_ticklabels(["{0:.0f}".format(n) for n in ticks / 10.**power ])
    cbar.set_label(rf"$dN / d\Omega (\times 10^{{{power:.0f}}}\,\mathrm{{cm}}^{{-2}}\mathrm{{s}}^{{-1}}\mathrm{{sr}}^{{-1}})$")

    # add circle for on region
    ra = casc_spec.cascmap.m.geom.center_skydir.ra.value
    dec = casc_spec.cascmap.m.geom.center_skydir.dec.value
    c_on = wcsaxes.SphericalCircle([ra, dec] * u.deg,
                                   radius=on_radius,
                                   edgecolor='w',
                                   lw=2,
                                   ls='--',
                                   facecolor='none',
                                   transform=ax.get_transform('fk5'))
    ax.add_patch(c_on)
    if psf_file is not None:
        c_psf = wcsaxes.SphericalCircle([ra, dec] * u.deg,
                                       radius=r_psf,
                                       edgecolor='w',
                                       lw=2,
                                       ls='-',
                                       facecolor='none',
                                       transform=ax.get_transform('fk5'))
        logging.info(f"Fermi LAT PSF3 68% containment: {r_psf}")
        ax.add_patch(c_psf)

    ax.annotate(f"$\mathbf{{{src:s}}}$", xy=(0.05,0.95), va='top', color='w', xycoords="axes fraction", **effect_k)
    ax.annotate(r"$B={0[0]:.1f} \times 10^{{{0[1]:0n}}}$G".format(get_base_power(B)),
                xy=(0.05,0.9), va='top', xycoords="axes fraction", color='w', fontsize='small', **effect_k)
    ax.annotate(r"$t_\mathrm{{max}}=10^{{{0[1]:0n}}}$ yrs".format(get_base_power(config['global']['tmax'])),
                    xy=(0.05,0.85), va='top', color='w', xycoords="axes fraction", fontsize='small', **effect_k)

    fig.subplots_adjust(left=0.07, bottom=0.15, top=0.95, right=0.99)
    for form in ['png', 'pdf']:
        fig.savefig(f"plots/casc_map_Egt1GeV_{src:s}_b{B}_t{config['global']['tmax']:.0e}.{form:s}")
    plt.close("all")

    # the SED with the best fit

    if ax_sed is None:
        ax_sed=plt.gca()
    e_range = [1e-3, 10.] * u.TeV

    col_casc1 = plt.cm.Set3(0.42)
    col_casc = plt.cm.Set1(0.5)

    # plot rescaled flux points
    if not casc_model.parameters['bias'].value == 0.: 
        if flux_points is not None:
            fp = flux_points.table.copy()
            fp['e_ref'] *= 1. + casc_model.parameters['bias'].value
            fp['e_min'] *= 1. + casc_model.parameters['bias'].value
            fp['e_max'] *= 1. + casc_model.parameters['bias'].value
            fp_rescale = FluxPoints(fp)
            fp_rescale.plot(ax=ax_sed, energy_power=2., marker='o', mec=plt.cm.Set1(0.), color=plt.cm.Set1(0.), mfc='none')

    # primary
    bias_value = casc_model.copy().parameters['bias'].value
    casc_model.parameters['bias'].value = 0.

    # total model
    casc_model.spectral_model.add_primary = True
    casc_model.spectral_model.plot(ax=ax_sed, energy_range=e_range, energy_power=2, color=col_casc1, label='Total', zorder=1)
    casc_model.spectral_model.plot_error(ax=ax_sed, energy_range=e_range, energy_power=2, color=col_casc1, alpha=1., zorder=1)
    # point source
    obs_model = casc_model.spectral_model.intrinsic_spectral_model * casc_model.spectral_model.ebl

    obs_model.plot(ax=ax_sed,
                   energy_range=e_range,
                   energy_power=2,
                   label='Primary',
                   color=col_casc1,
                   ls='--',
                   zorder=1.1,
                   **effect)


    # cascade
    casc_model.spectral_model.add_primary = False
    casc_model.spectral_model.plot(ax=ax_sed,
                                   energy_range=e_range,
                                   energy_power=2,
                                   label='Cascade',
                                   color=col_casc1,
                                   zorder=-1,
                                   ls='-.', **effect)
    casc_model.spectral_model.add_primary = True

    # reset bias
    casc_model.parameters['bias'].value = bias_value

    ax_sed.legend(loc="best", fontsize='small')

    energy_unit = 'TeV'
    ax_sed.set_xlabel('Energy ({0:s})'.format(energy_unit))
    ax_sed.set_ylabel('$E^{{2}} dN/dE$ ({0:s} cm$^{{-2}}$ s$^{{-1}}$)'.format(energy_unit))
    #plt.savefig(f"plots/casc_fit.png")

    return

def plot_best_fit_bias(b_fields, best_fit_bias, fig=None):
    """
    Plot the best-fit bias values
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 2 * len(b_fields))) 

    for i, (src, pars) in enumerate(best_fit_bias.items()):
        ax = fig.add_subplot(len(b_fields), 1, i + 1)
        ax.errorbar(b_fields, [p.value for p in pars], yerr=[p.error for p in pars], ls="none", marker="o", label=src, color=f"C{i}")
        if not i == len(b_fields) -1:
            ax.tick_params(labelbottom=False)
        ax.set_xscale("log")
        ax.legend(loc=1, fontsize='small')
        ax.axhline(0.2, ls='--', color='0.5')
        ax.axhline(-0.2, ls='--', color='0.5')
        ax.grid()
        
    ax.set_xlabel("$B$ field (G)")
    ax.set_ylabel("bias", y=len(b_fields) / 2)
    fig.subplots_adjust(wspace=0., hspace=0., top=0.95, bottom=0.01)
    fig.suptitle("$t_\mathrm{{max}}=10^{{{0[1]:0n}}}$ yrs".format(get_base_power(config['global']['tmax'])))

    return fig

if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the combined fermi and HESS fit"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('--select-source')
    parser.add_argument('--select-bfield', type=float)
    parser.add_argument('--plots', action="store_true", help="Create plots")
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

    # set up outdir 
    outdir = os.path.join(config['global']['outdir'], "tmax{0:.1e}".format(config['global']['tmax']))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # plot overall results
    try:
        from simCRpropa.scripts.combined_fit_1d import plot_likelihood_vs_B
        ax = plot_likelihood_vs_B(config, simple_title=True, skip_sources=['1RXSJ195815.6-301119'], cmap=plt.cm.Dark2)
        for form in ['png', 'pdf']:
            plt.savefig(f"plots/lnl_vs_B_t{config['global']['tmax']:.0e}_bias{config['global']['fix_bias']:n}.{form:s}", dpi=150)
        plt.close("all")

    except ImportError:
        pass

    best_fit_bias = {}

    # loop through the sources:
    for src in config.keys():
        if src == 'global':
            continue

        if args.select_source is not None:
            if not src == args.select_source:
                continue

        best_fit_bias[src] = []

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
            dataset_stack = Datasets.read(dataset_1d_file.replace(".yaml", "_stacked.yaml"), 
                                         )
            models = Models.read(dataset_1d_file.replace("_datasets.yaml", "_stacked_pl_model.yaml"))
            models[0].datasets_names = dataset_stack.names
            dataset_stack.models = models


        else:
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

        # Define the model for the intrinsic / observed blazar spectrum
        epl = ExpCutoffPowerLawSpectralModel(
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            index=1.6 * u.dimensionless_unscaled,
            lambda_=0.1 * u.Unit("TeV-1"),
            reference=1 * u.TeV,
        )

        ebl = EBLAbsorptionNormSpectralModel.read(filename=config['global']['gammapy_ebl_file'],
                                                  redshift=config[src]['z_src'])

        obs = epl * ebl
        ps_model = SkyModel(spectral_model=obs.copy(), name='ps')

        # loop over magnetic fields
        save_plot_obs = True
        for ib, B in enumerate(b_fields):
            if args.select_bfield is not None:
                base_B1, power_B1 = get_base_power(B)
                base_B2, power_B2 = get_base_power(B)
                
                if not (np.round(base_B1,2) == np.round(base_B2, 2) and power_B1 == power_B2):
                    continue

            ds = dataset_stack[0].copy()
            pars_out_file = os.path.join(outdir, f"best_fit_pars_{src:s}_B{B:.2e}_fix_bias{config['global']['fix_bias']}.fits")
            parameters = Table.read(pars_out_file)
            par_list = []
            par_list = []
            for row in parameters:
                row = dict(**row)
                row['frozen'] = bool(row['frozen'])
                par_list.append(Parameter(**row))

            parameters = Parameters(par_list)
            best_fit_bias[src].append(parameters['bias'])

            fig_sed, ax_sed, flux_points = plot_obs_sed(ds, sed, plot_fermi_fp=True)

            ax_sed.annotate(f"$\mathbf{{{src:s}}}$", xy=(0.05,0.95), va='top', xycoords="axes fraction", **effect)

            fig_sed.tight_layout()
            if save_plot_obs:
                for form in ['png', 'pdf']:
                    fig_sed.savefig(f"plots/{src:s}_obs.{form:s}")
                

            ax_sed.annotate(r"$B={0[0]:.1f} \times 10^{{{0[1]:0n}}}$G".format(get_base_power(B)),
                            xy=(0.05,0.88), va='top', xycoords="axes fraction", fontsize='small', **effect)
            ax_sed.annotate(r"$t_\mathrm{{max}}=10^{{{0[1]:0n}}}$ yrs".format(get_base_power(config['global']['tmax'])),
                            xy=(0.05,0.83), va='top', xycoords="axes fraction", fontsize='small', **effect)

            plot_best_fit_cascade(config, geom, ps_model, ds, parameters,
                                  B=B, on_radius=on_radius, ax_sed=ax_sed, 
                                  psf_file=os.path.join(os.path.dirname(sed_file), "psf_01.fits"), 
                                  flux_points=flux_points
                                  )

            for form in ['png', 'pdf']:
                fig_sed.savefig(f"plots/{src:s}_obs_casc_B{B:.2e}_t{config['global']['tmax']:.0e}_bias{config['global']['fix_bias']:n}.{form:s}")
            save_plot_obs = False

    if config['global']['fix_bias'] is False:
        fig = plot_best_fit_bias(b_fields, best_fit_bias)
        fig.savefig(f"plots/best_fit_bias_t{config['global']['tmax']:.0e}.png", dpi=150)
        plt.close("all")

# TODO
#  - make sure that bias implementation is correct
