import numpy as np
from astropy import units as u


def integral_sim_pl(config, E0=1e12):
    """
    Calculate the integral for a power-law spectrum analytically
    using the values stored in config['Source']j

    :param config: dict
        Dictionary for CRPropa simulation
    :param E0: float
        Energy scale of power law in eV, default: 1e12 eV

    :return: float
        the integral of the power law over energy
    """
    # solving integral analytically
    # for gamma_sim = -1
    if config['Source']['index'] == -1:
        result = np.log(config['Source']['Emax'] / E0) - np.log(config['Source']['Emin'] / E0)
        result *= E0

    # for other gamma_sim values
    else:
        gamma_p_1 = config['Source']['index'] + 1.
        result = config['Source']['Emax'] ** gamma_p_1 - config['Source']['Emin'] ** gamma_p_1
        result /= E0 ** config['Source']['index']
        result /= gamma_p_1

    # unit is in eV
    return result


def get_weights(injected_energies, target_spectral_shape, config, E0=1e12):
    """
    Compute spectral weights for individual injected photons for re-weighting cascade histogram.
    For a target spectrum

    $$ F_\mathrm{target}(E) = N_0 f(E/E_0), $$

    where f(E/E_0) is the spectral shape, the weights are computed as

    $$ w_i = \frac{f(E_i/E_0)}{(E_i/E_0)^{\gamma_\mathrm{sim}}} $$

    For each injected energy $E_i$.

    :param injected_energies: array-like
        injected energies (E0 entry in CRPropa output) in eV

    :param target_spectral_shape: function
        function that returns the spectral shape, i.e., the spectrum without normalization.
        Function needs to be able to accept arrays with energies in eV

    :param config: dict
        Dictionary for CRPropa simulation

    :param E0: float
        Energy scale of power law in eV, default: 1e12

    :return:
    Array with spectral weights as numpy array


    """

    weight = target_spectral_shape(injected_energies).to(u.dimensionless_unscaled).value
    weight /= (injected_energies.to('eV').value / E0)**config['Source']['index']

    return weight


def build_casc_histogram(data, energy_edges, weights=None, tmax_years=1e7):
    """
    Build the histogram of observed cascade photons

    :param data: dict
        output dictionary from CRPropa simulation

    :param energy_edges:
        The energy edges in eV for the histogram

    :param weights: array-like
        The spectral weights computed with the get_weight function

    :param tmax_years: float
        Maximum of source activity time in years

    :return: dict
        dictionary with results:
            - counts: the counts in each bin
            - Ecen: the central energies of each observed energy bin
            - dE: the bin width
    """
    mask = (data['ID1'] == 11) | (data['ID1'] == -11)  # true for casc photons

    mask_tdelay = mask & (data['dt'] <= tmax_years)

    mask_tot = mask_tdelay * data['mask']

    result = {}

    if weights is not None:
        weights = weights[mask_tot]

    result['counts'], bins = np.histogram(data['E'][mask_tot],
                                          bins=energy_edges, weights=weights
                                          )

    result['Ecen'] = np.sqrt(bins[1:] * bins[:-1])
    result['dE'] = bins[1:] - bins[:-1]

    return result


def build_casc_histogram_individual_bin(data, energy_edges, injected_energy, weights=None, tmax_years=1e7):
    """
    Build the histogram of observed cascade photons for one injected energy

    :param data: dict
        output dictionary from CRPropa simulation

    :param energy_edges:
        The energy edges in eV for the histogram

    :param injected_energy: float
        injected energy in eV

    :param weights: array-like
        The spectral weights computed with the get_weight function

    :param tmax_years: float
        Maximum of source activity time in years

    :return: dict
        dictionary with results:
            - counts: the counts in each bin
            - Ecen: the central energies of each observed energy bin
            - dE: the bin width
    """
    mask = (data['ID1'] == 11) | (data['ID1'] == -11)  # true for casc photons

    mask_tdelay = mask & (data['dt'] <= tmax_years)

    mask_tot = mask_tdelay * data['mask']

    # only one energy bin:
    mask_tot &= (data['E0'] == injected_energy)

    if not np.sum(mask_tot):
        print(f"No cascade photons for {injected_energy / 1e9:8.2f} GeV injected energy")
        return None

    result = {}

    if weights is not None:
        weights = weights[mask_tot]

    result['counts'], bins = np.histogram(data['E'][mask_tot],
                                          bins=energy_edges, weights=weights
                                          )

    result['Ecen'] = np.sqrt(bins[1:] * bins[:-1])
    result['dE'] = bins[1:] - bins[:-1]

    return result

def build_casc_spectrum(data, config, target_spectral_shape, params,
                        energy_edges, tmax_years=1e7):
    """
    Compute the cascade spectrum from a simulation that used a power law as input
    and not individual energies.

    :param data: dict
        output dictionary from CRPropa simulation

    :param config: dict
        Dictionary for CRPropa simulation

    :param target_spectral_shape: function
        function that returns the spectral shape, i.e., the spectrum without normalization.
        Function needs to be able to accept arrays with energies in eV

    :param params: dict
        parameters of target injected spectrum

    :param energy_edges:
        The energy edges in eV for the histogram

    :param tmax_years: float
        Maximum of source activity time in years

    :return: dict
        dictionary with results:
            - counts: the counts in each bin
            - Ecen: the central energies of each observed energy bin
            - dE: the bin width
            - dnde: the cascade spectrum in physical units
    """

    weights = get_weights(data['E0'] * u.eV, target_spectral_shape, config, E0=params['Scale'].to('eV').value)

    casc_hist_w = build_casc_histogram(data,
                                       energy_edges=energy_edges,
                                       weights=weights,
                                       tmax_years=tmax_years)

    integral_sim = integral_sim_pl(config, E0=params['Scale'].to('eV').value) * u.eV

    # number of simulated particles
    n_sim = config['Simulation']['Nbatch'] * config['Simulation']['multiplicity']

    casc_hist_w['dnde'] = casc_hist_w['counts'] / (casc_hist_w['dE'] * u.eV)
    casc_hist_w['dnde'] *= params['Prefactor'] * integral_sim / n_sim

    return casc_hist_w


def build_casc_spectrum_bin_by_bin(data, target_spectral_shape, config, params,
                                   energy_edges, tmax_years=1e7):
    """
    Compute the cascade spectrum from a simulation that used individual energies as injectio
    and not a specific spectrum.

    :param data: dict
        output dictionary from CRPropa simulation

    :param config: dict
        Dictionary for CRPropa simulation

    :param target_spectral_shape: function
        function that returns the spectral shape, i.e., the spectrum without normalization.
        Function needs to be able to accept arrays with energies in eV

    :param params: dict
        parameters of target injected spectrum

    :param energy_edges:
        The energy edges in eV for the histogram

    :param tmax_years: float
        Maximum of source activity time in years

    :return: dict
        dictionary with results:
            - counts: the counts in each bin
            - Ecen: the central energies of each observed energy bin
            - dE: the bin width
            - dnde: the cascade spectrum in physical units
    """
    injected_energies = np.unique(data['E0'])

    tot_hist_w = np.zeros(energy_edges.size - 1)

    _weight = []

    for i, E_inj in enumerate(injected_energies):

        # this is the weight for a gamma = -1 assumed
        # injected power law which is equivalent to individual injected energies
        weight = target_spectral_shape(E_inj * u.eV)
        weight *= E_inj
        weight *= np.log(config['Source']['Emax'][i] / config['Source']['Emin'][i])
        weight = weight.to(u.dimensionless_unscaled).value

        casc_individual = build_casc_histogram_individual_bin(data, energy_edges, E_inj,
                                                              weights=None,
                                                              tmax_years=tmax_years)

        _weight.append(weight * params['Prefactor'])
        # no cascade photons for this energy
        if casc_individual is None:
            continue

        n_sim = config['Simulation']['Nbatch'][i] * config['Simulation']['multiplicity']
        tot_hist_w += casc_individual['counts'] * weight / n_sim

    tot_hist_w = tot_hist_w * params['Prefactor'] * u.eV / (casc_individual['dE'] * u.eV)

    casc_bin_by_bin = dict(dnde=tot_hist_w)
    casc_bin_by_bin['dE'] = casc_individual['dE']
    casc_bin_by_bin['Ecen'] = casc_individual['Ecen']

    #print(_weight)

    return casc_bin_by_bin

