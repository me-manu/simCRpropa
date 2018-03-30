from crpropa import *
import logging
import yaml
import numpy as np
import argparse
from os import path
from haloanalysis.batchfarm import utils,lsf
from copy import deepcopy
from glob import glob
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
import simCRpropa
from simCRpropa import collect
from collections import OrderedDict
import h5py

def initRandomField(vgrid, Bamplitude, seed = 0):
    np.random.seed(seed)
    gridArray = vgrid.getGrid()
    nx = vgrid.getNx()
    ny = vgrid.getNy()
    nz = vgrid.getNz()
    logging.info("vgrid: nx = {0:n}, ny = {0:n}, nz = {0:n}".format(
        nx,ny,nz))
    for xi in xrange(0,nx):
        for yi in xrange(0,ny):
            for zi in xrange(0,nz):
                vect3d = vgrid.get(xi,yi,zi)

                x = np.random.uniform(-1,1)
                y = np.random.uniform(-1,1)
                z = np.random.uniform(-1,1)
                d = np.sqrt(x*x+y*y+z*z)
                #while d > 1:  
                #    x = np.random.uniform(-1,1)
                #    y = np.random.uniform(-1,1)
                #    z = np.random.uniform(-1,1)
                #    d = np.sqrt(x*x+y*y+z*z)

                vect3d.x = Bamplitude * x/d
                vect3d.y = Bamplitude * y/d
                vect3d.z = Bamplitude * z/d
    return None

def build_histogram(combined, config, cascparent = 11., intparent = 22., obs = 22.):
    """
    Build a numpy histogram for the cascade 
    and intrinsic spectrum

    Parameters
    ----------
    combined: `~h5py.File`
        combined hdf5 histogram
    config: dict
        dict with configuration

    {options}
    cascparent: float
        parent particle ID for cascade
    intparent: float
        parent particle ID for intrinsic spectrum
    obs: float
        particle ID for observed spectrum

    Returns
    -------
    tuple with instrinsic spectrum, cascade spectrum, and 
    energy bins
    """
    Ebins = np.logspace(np.log10(config['Source']['Emin']),
                 np.log10(config['Source']['Emax']),
                                 config['Source']['Esteps'])
    Ecen = np.sqrt(Ebins[1:] * Ebins[:-1])

    # intrinsic spectrum
    intspec = np.zeros((Ecen.size,Ecen.size))
    # casc spectrum
    casc = np.zeros((Ecen.size,Ecen.size))

    for i in range(Ebins.size - 1):
        m = (combined['simEM/ID1/Ebin{0:03n}'.format(i)][...] == np.abs(intparent)) \
                & (combined['simEM/ID/Ebin{0:03n}'.format(i)][...] == np.abs(obs))
        h = np.histogram(combined['simEM/E/Ebin{0:03n}'.format(i)][m], bins = Ebins)
        intspec[i,:] = h[0]
                                
        m = (combined['simEM/ID1/Ebin{0:03n}'.format(i)][...] == np.abs(cascparent)) \
                & (combined['simEM/ID/Ebin{0:03n}'.format(i)][...] == np.abs(obs))
        h = np.histogram(combined['simEM/E/Ebin{0:03n}'.format(i)][m], bins = Ebins)
        casc[i,:] = h[0]
    return intspec, casc, Ebins

def build_histogram_obs(combined, config, obs = 22., Ebins = np.array([])):
    """
    Build a numpy histogram for the cascade spectrum 
    of some particle type. 

    Parameters
    ----------
    combined: `~h5py.File`
        combined hdf5 histogram
    config: dict
        dict with configuration

    {options}
    obs: float 
        particle ID for observed spectrum
    Ebins: `~numpy.ndarray`
        custom energy binning.
        If zero length, use binning from config file

    Returns
    -------
    `~numpy.ndarray` with cascade spectrum
    """
    if Ebins.size == 0:
        Ebins = np.logspace(np.log10(config['Source']['Emin']),
                 np.log10(config['Source']['Emax']),
                                 config['Source']['Esteps'])
    Ecen = np.sqrt(Ebins[1:] * Ebins[:-1])

    # casc spectrum
    casc = np.zeros((config['Source']['Esteps'] - 1,Ecen.size))

    for i,k in enumerate(combined['simEM/E'].keys()):
                                
        m = combined['simEM/ID/{0:s}'.format(k)][...] == np.abs(obs)
        h = np.histogram(combined['simEM/E/{0:s}'.format(k)][m], bins = Ebins)
        casc[i,:] = h[0]
    return casc, Ebins

defaults = """
FileIO:
    outdir: ./

Simulation:
    Nparticle: 1.e+6 # total number of particles to simulate
    Nbatch: 1.e+4 # number of particles simulated in each simulation
    cpu_n: 8

Source:
    z: 0.14 # source redshift, source position is at (D,0,0)
    Energy: 1.e+13 # energy of single particle in eV
    th_jet: 5. # jet opening angle in degrees, emission in -x direction
    Emin: 100.e+9 # minimum source energy if spectrum is used, in eV
    Emax: 20.e+12 # maximum source energy if spectrum is used, in eV
    Esteps: 20 # maximum source energy if spectrum is used, in eV
    index: -1.5 # source spectral index
    cutoff: 10. #cutoff energy in TeV
    useSpectrum: True # if True, simulate spectrum, otherwise inject delta function with energy E
    Spectrum: "E^{0[index]:.3f}*exp(-E/({0[cutoff]:.2f} * TeV))"

Observer:
    obsPosX: 0. # observer's x coordinate
    obsPosY: 0. # observer's y coordinate
    obsPosZ: 0. # observer's z coordinate
    obsAngle: 0. # angle w.r.t. to jet axis, if non-zero, overwrites source position
    th_jet: 5. # jet opening angle in degrees, emission in -x direction
    zmin: 1.e-3 # minimum redshift for detection ~ 14 Mpc
    obsSize: 10. # size of observer's small sphere in Mpc
    obsSmallSphere: True # use small sphere observer, otherwise use large sphere observer

BreakConditions:
    Dmax: 1. # maximum distance of particle for tracing
    Emin: 1.e+9 # minimum energy for particle tracing

Bfield:
    B: 1.e-16 # rms of B field in Gauss
    seed: 2308 # random seed for B field
    type: cell # either cell or turbulence
    NBgrid: 200 # number of B-field cells (turbulent field only)
    boxSize: 100 # box size for vector grid in Mpc (turbulent field only)
    maxTurbScale: 50. # maximum turbulence scale in Mpc / cell size for cell-like field
    turbIndex: -3.667 # turbulence index, default - 11/3
    periodicity: 4000. # length of periodic box in Mpc
    EBL: IRB_Gilmore12
"""

class SimCRPropa(object):
    def __init__(self, **kwargs):
        """
        Initialize the class
        """
        df = yaml.load(defaults)
        for k,v in df.items():
            kwargs.setdefault(k,v)
            for kk,vv in v.items():
                kwargs[k].setdefault(kk,vv)
        self.config = deepcopy(kwargs)
        self.__dict__.update(self.config)

        self.D = redshift2ComovingDistance(self.Source['z']) # comoving source distance
        self.emcasc = self.Simulation['emcasc']

        if type(self.Bfield['B']) == list:
            self._bList = deepcopy(self.Bfield['B'])
            self.Bfield['B'] = self._bList[0]
        elif type(self.Bfield['B']) == float:
            self._bList = [self.Bfield['B']]
        else:
            raise Exception("B type not understood: {0}".format(
                type(self.Bfield['maxTurbScale'])))

        if type(self.Bfield['maxTurbScale']) == list:
            self._turbScaleList = deepcopy(self.Bfield['maxTurbScale'])
            self.Bfield['maxTurbScale'] = self._turbScaleList[0]
        elif type(self.Bfield['maxTurbScale']) == float:
            self._turbScaleList = [self.Bfield['maxTurbScale']]
        else:
            raise Exception("maxTurbScale type not understood: {0}".format(
                type(self.Bfield['maxTurbScale'])))

        if self.Bfield['EBL'] == 'IRB_Gilmore12':
            self._EBL = IRB_Gilmore12 #Dominguez11, Finke10, Franceschini08
        elif self.Bfield['EBL'] == 'Dominguez11':
            self._EBL = Dominguez11
        elif self.Bfield['EBL'] == 'Finke10':
            self._EBL = Finke10
        elif self.Bfield['EBL'] == 'Franceschini08':
            self._EBL = Franceschini08
        else:
            raise ValueError("Unknown EBL model chosen")
        self._URB = URB_Protheroe96
        return

    def setOutput(self,jobid, idB = 0, idL = 0):
        """Set output file and directory"""
        self.OutName = 'casc_{0:05n}.dat'.format(jobid)

        # append options to file path
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['basedir'],
                        'z{0[z]:.3f}/th_jet{0[th_jet]}/'.format(self.Source)))
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        'th_obs{0[obsAngle]}/'.format(self.Observer)))
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        'spec{0[useSpectrum]:n}/'.format(self.Source)))

        self.Bfield['B'] = self._bList[idB]
        self.Bfield['maxTurbScale'] = self._turbScaleList[idL]

        if self.Bfield['type'] == 'turbulence':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                'Bturb{0[B]:.2e}/q{0[turbIndex]:.2f}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield)))
        elif self.Bfield['type'] =='cell':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                'Bcell{0[B]:.2e}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield)))
        else:
            raise ValueError("Bfield type must be either 'cell' or 'turbulence' not {0[type]}".format(self.Bfield))

        self.outputfile = path.join(self.FileIO['outdir'],self.OutName)
        logging.info("outdir: {0[outdir]:s}".format(self.FileIO))
        logging.info("outfile: {0:s}".format(self.outputfile))
        return

    def _create_bfield(self):
        """Set up simulation volume and magnetic field"""
        boxOrigin = Vector3d(0, 0, 0)
        boxSpacing = self.Bfield['boxSize'] * Mpc / self.Bfield['NBgrid']

        if self.Bfield['type'] == 'turbulence':
            logging.info('Box spacing for B field: {0:.3e} Mpc'.format(boxSpacing / Mpc))
            vgrid = VectorGrid(boxOrigin, self.Bfield['NBgrid'], boxSpacing)
            initTurbulence(vgrid, self.Bfield['B'] * gauss, 
                            2 * boxSpacing, 
                            self.Bfield['maxTurbScale'] * Mpc, 
                            self.Bfield['turbIndex'],
                            self.Bfield['seed'])
            bField0 = MagneticFieldGrid(vgrid)
            self.bField = PeriodicMagneticField(bField0,
                    Vector3d(self.Bfield['periodicity']* Mpc), Vector3d(0), False)
            self.__extent = self.Bfield['boxSize'] * Mpc
            logging.info('B field initialized')
            logging.info('Lc = {0:.3e} kpc'.format(
                turbulentCorrelationLength(2. * boxSpacing / Mpc * 1e3,
                                        self.Bfield['maxTurbScale'] * 1e3,
                                        self.Bfield['turbIndex'])))  # correlation length, input in kpc

        if self.Bfield['type'] == 'cell':
            logging.info('Box spacing for cell-like B field: {0:.3e} Mpc'.format(self.Bfield['maxTurbScale']))
            vgrid = VectorGrid(boxOrigin,
                                int(np.ceil(redshift2ComovingDistance(self.Source['z'])/\
                                            self.Bfield['maxTurbScale'] / Mpc)),
                                self.Bfield['maxTurbScale'] * Mpc)
            initRandomField(vgrid, self.Bfield['B'] * gauss, seed = self.Bfield['seed'])
            self.bField = MagneticFieldGrid(vgrid)
            self.__extent = int(np.ceil(redshift2ComovingDistance(self.Source['z'])/\
                                    self.Bfield['maxTurbScale'] / Mpc)) \
                                    * self.Bfield['maxTurbScale'] * Mpc
            logging.info('B field initialized')


        logging.info('vgrid extension: {0:.3e} Mpc'.format(self.__extent / Mpc))
        logging.info('<B^2> = {0:.3e} nG'.format((rmsFieldStrength(vgrid) / nG)))   # RMS
        logging.info('<|B|> = {0:.3e} nG'.format((meanFieldStrength(vgrid) / nG)))  # mean
        logging.info('B(10 Mpc, 0, 0)={0} nG'.format(self.bField.getField(Vector3d(10,0,0) * Mpc) / nG))
        return

    def _create_observer(self):
        """Set up the observer for the simulation"""
        obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
        self.observer = Observer()
        #ObserverSmallSphere (Vector3d center=Vector3d(0.), double radius=0)
        #Detects particles upon entering a sphere
        if self.Observer['obsSmallSphere']:
            self.observer.add(ObserverSmallSphere(obsPosition, self.Observer['obsSize'] * Mpc))
        else:
            # also possible: detect particles upon exiting a shpere: 
            # ObserverLargeSphere (Vector3d center=Vector3d(0.), double radius=0)
            # radius is of large sphere is equal to source distance
            self.observer.add(ObserverLargeSphere(obsPosition, self.D))
        # looses a lot of particles -- need periodic boxes
        #Detects particles in a given redshift window. 
        #self.observer.add(ObserverRedshiftWindow(-1. * self.Observer['zmin'], self.Observer['zmin']))
        self.observer.add(ObserverElectronVeto())

        # for CR secondaries testing
        self.observer.add(ObserverNucleusVeto())
        #ObserverNucleusVeto
        #ObserverTimeEvolution

        self.output = TextOutput(self.outputfile,
                                    Output.Event3D)

        self.output.enable(Output.CurrentIdColumn)
        self.output.enable(Output.CurrentDirectionColumn)
        self.output.enable(Output.CurrentEnergyColumn)
        self.output.enable(Output.CurrentPositionColumn)
        self.output.enable(Output.CreatedIdColumn)
        self.output.enable(Output.SourceEnergyColumn)
        self.output.enable(Output.TrajectoryLengthColumn)
        self.output.enable(Output.SourceDirectionColumn)

        self.output.disable(Output.RedshiftColumn)
        self.output.disable(Output.CreatedDirectionColumn)
        self.output.disable(Output.CreatedEnergyColumn)
        self.output.disable(Output.CreatedPositionColumn)
        self.output.disable(Output.SourceIdColumn)
        self.output.disable(Output.SourcePositionColumn)


        logging.info('Saving output to {0:s}'.format(self.outputfile))
        self.output.setEnergyScale(eV)
        self.observer.onDetection(self.output)

        logging.info('Observer and output initialized')
        return

    def _create_source(self):
        """Set up the source for the simulation"""
        self.source = Source()
        self.source.add(SourceRedshift(self.Source['z']))
        if self.Observer['obsSmallSphere']:
            self.source.add(SourcePosition(Vector3d(self.D, 0, 0)))
            # emission to negativ x-axis
            self.source.add(SourceEmissionCone(
            Vector3d(np.cos(np.pi - np.radians(self.Observer['obsAngle'])), 
                np.sin(np.pi - np.radians(self.Observer['obsAngle'])), 0), 
                np.radians(self.Source['th_jet'])))
        else:
            obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
            # obs position same as source position for LargeSphere Observer
            self.source.add(SourcePosition(obsPosition))
            # emission cone towards positiv x-axis
            self.source.add(SourceEmissionCone(
                Vector3d(np.cos(np.radians(self.Observer['obsAngle'])), 
                    np.sin(np.radians(self.Observer['obsAngle'])), 0), 
                np.radians(self.Source['th_jet'])))
        # SourceParticleType takes int for particle ID. 
        # for a nucleus with A,Z you can use nucleusId(int a, int z) function
        # other IDs are given in http://pdg.lbl.gov/2016/reviews/rpp2016-rev-monte-carlo-numbering.pdf
        # e- : 11, e+ -11 ; antiparticles have negative sign
        # nu_e : 12
        # mu : 13
        # nu_mu : 14
        # nu_tau : 16
        # proton: 2212
        if self.Source['useSpectrum']:
            spec = self.Source['Spectrum'].format(self.Source)
            logging.info('Spectrum: {0}'.format(spec))
            genericSourceComposition = SourceGenericComposition(self.Source['Emin'] * eV, 
                                                                self.Source['Emax'] * eV, 
                                                                spec)
            genericSourceComposition.add(self.Source['Composition'],1)
            self.source.add(genericSourceComposition)
            # for a power law use SourcePowerLawSpectrum (double Emin, double Emax, double index)
        else:
        # mono-energetic particle:
            self.source.add(SourceParticleType(self.Source['Composition']))
            self.source.add(SourceEnergy(self.Source['Energy'] * eV))
        logging.info('source initialized')
        return

    def _setup_emcascade(self):
        """Setup simulation module for electromagnetic cascade"""
        self.m = ModuleList()
        #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4, double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
        #self.m.add(PropagationCK(self.bField, 1e-2, 100 * kpc, 10 * Mpc))
        self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        #m.add(Redshift())
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        # Extends to negative redshift values to allow for symmetric time windows around z=0
        self.m.add(FutureRedshift())
        self.m.add(EMInverseComptonScattering(CMB, True))
        self.m.add(EMInverseComptonScattering(self._EBL, True))
        # EMPairProduction:  electron-pair production of cosmic ray photons 
        #with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
        # EMPairProduction(PhotonField photonField = CMB, bool haveElectrons = false,double limit = 0.1 ), 
        #if haveElectrons = true, electron positron pair is created
        # EMInverComptonScattering(PhotonField photonField = CMB,bool havePhotons = false,double limit = 0.1 ), 
        #if havePhotons = True, photons are created
        # also availableL EMDoublePairProduction, EMTripletPairProduction
        self.m.add(EMPairProduction(self._EBL, True))
        self.m.add(EMPairProduction(CMB, True))
        # for photo-pion production: 
        #PhotoPionProduction (PhotonField photonField=CMB, bool photons=false, bool neutrinos=false, 
        # bool antiNucleons=false, double limit=0.1, bool haveRedshiftDependence=false)
        # for photo disentigration:
        #PhotoDisintegration (PhotonField photonField=CMB, bool havePhotons=false, double limit=0.1)
        # for nuclear decay:
        #NuclearDecay (bool electrons=false, bool photons=false, bool neutrinos=false, double limit=0.1)
        # Synchrotron radiation: 
        #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
        #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
        #Large number of particles can cause memory problems!
        self.m.add(SynchrotronRadiation(self.bField, True))
        logging.info('modules initialized')
        return

    def _setup_crcascade(self):
        """
        Setup simulation module for cascade initiated by cosmic rays
        
        kwargs 
        ------
        """
        if self.emcasc:
            photons = True
            electrons = True
            neutrinos = False
            #thinning = 0.9
            thinning = 0.95
        else:
            photons = False
            electrons = False
            neutrinos = True
            thinning = 0.
        antinucleons = False if self.Source['Composition'] == 2212 else True
        limit = 0.1 # limit to 0.5 instead 0.1 due to memory at high energies
                    # of step size limit as fraction of mean free path 

        self.m = ModuleList()
        #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4,
        #double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
        #self.m.add(PropagationCK(self.bField, 1e-2, 100 * kpc, 10 * Mpc))
        #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
        if self.Source['Energy'] >= 1e18 and self.emcasc:
            logging.info("Energy is greater than 1 EeV, limiting " \
                        "sensitivity due to memory. E = {0[Energy]:.3e}".format(self.Source))
            #self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))
            self.m.add(PropagationCK(self.bField, 1e-4, 10. * kpc, 10 * Mpc))
        else:
            self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))
            # this takes about a factor of five longer:
            #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        #m.add(Redshift())
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        # Extends to negative redshift values to allow for symmetric time windows around z=0
        self.m.add(FutureRedshift())
        if self.emcasc:
            #self.m.add(EMInverseComptonScattering(CMB, photons, limit)) # not activated in example notebook
            #self.m.add(EMInverseComptonScattering(self._EBL, photons, limit)) # not activated in example notebook
            # EMPairProduction:  electron-pair production of cosmic ray photons 
            #with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
            # EMPairProduction(PhotonField photonField = CMB, bool haveElectrons = false,double limit = 0.1 ), 
            #if haveElectrons = true, electron positron pair is created
            # EMInverComptonScattering(PhotonField photonField = CMB,bool havePhotons = false,double limit = 0.1 ), 
            #if havePhotons = True, photons are created
            # also availableL EMDoublePairProduction, EMTripletPairProduction
            #self.m.add(EMPairProduction(self._EBL, electrons, limit)) # not activated in example notebook

            #self.m.add(EMPairProduction(CMB, electrons, limit)) # not activated in example notebook

            self.m.add(EMInverseComptonScattering(CMB, photons, thinning))
            self.m.add(EMInverseComptonScattering(self._URB, photons, thinning))
            self.m.add(EMInverseComptonScattering(self._EBL, photons, thinning))

            self.m.add(EMPairProduction(CMB, electrons, thinning))
            self.m.add(EMPairProduction(self._URB, electrons, thinning))
            self.m.add(EMPairProduction(self._EBL, electrons, thinning))
            self.m.add(EMDoublePairProduction(CMB, electrons, thinning))

            self.m.add(EMDoublePairProduction(self._URB, electrons, thinning))
            self.m.add(EMDoublePairProduction(self._EBL, electrons, thinning))

            self.m.add(EMTripletPairProduction(CMB, electrons, thinning))
            self.m.add(EMTripletPairProduction(self._URB, electrons, thinning))
            self.m.add(EMTripletPairProduction(self._EBL, electrons, thinning))

        # for photo-pion production: 
        # PhotoPionProduction (PhotonField photonField=CMB, bool photons=false, bool neutrinos=false, 
        # bool antiNucleons=false, double limit=0.1, bool haveRedshiftDependence=false)
        self.m.add(PhotoPionProduction(CMB, photons, neutrinos, antinucleons, limit, True))
        self.m.add(PhotoPionProduction(self._EBL, photons, neutrinos, antinucleons, limit, True))

        # ElectronPairProduction (PhotonField photonField=CMB, bool haveElectrons=false, double limit=0.1)
        # Electron-pair production of charged nuclei with background photons. 
        self.m.add(ElectronPairProduction(CMB, electrons, limit))
        self.m.add(ElectronPairProduction(self._EBL, electrons, limit))
        if not self.Source['Composition'] == 2212: # protons don't decay or diseintegrate
            # for nuclear decay:
            #NuclearDecay (bool electrons=false, bool photons=false, bool neutrinos=false, double limit=0.1)
            self.m.add(NuclearDecay(electrons, photons, neutrinos))
            # for photo disentigration:
            #PhotoDisintegration (PhotonField photonField=CMB, bool havePhotons=false, double limit=0.1)
            self.m.add(PhotoDisintegration(CMB, photons))
            self.m.add(PhotoDisintegration(self._EBL, photons))
        # Synchrotron radiation: 
        #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
        #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
        #Large number of particles can cause memory problems!
        #self.m.add(SynchrotronRadiation(self.bField, photons)) # not in example notebook
        logging.info('modules initialized')
        return

    def _setup_break(self):
        """Setup breaking conditions"""
        # add breaking conditions
        self.m.add(MinimumEnergy(self.BreakConditions['Emin'] * eV))
        self.m.add(self.observer)
        # stop tracing particle once it's propagation is longer than Dmax
        if self.D / Mpc > 100.:
            dmax = np.min([self.BreakConditions['Dmax'] * 1000.,self.D * 1.5 / Mpc])
        else: 
            dmax = self.BreakConditions['Dmax'] * 1000.
        self.m.add(MaximumTrajectoryLength(dmax * Mpc)) # Dmax is COMOVING
        # deactivate particle below a certain redshift
        self.m.add(MinimumRedshift(-1. * self.Observer['zmin']))

        # periodic boundaries
        #self.extent is the size of the B field grid        
        #sim.add(PeriodicBox(Vector3d(-self.__extent), Vector3d(2 * self.__extent)))
        logging.info('breaking conditions initialized')
        return

    def setup(self):
        """Setup the simulation"""
        self._create_bfield()
        if self.Source['Composition'] == 22 or \
            self.Source['Composition'] == 11 or \
            self.Source['Composition'] == -11:
            self._setup_emcascade()
        else:
            self._setup_crcascade()
        self._create_source()
        self._create_observer()
        self._setup_break()
        return 

    @lsf.setLsf
    def run(self,  overwrite = False, force_combine = False,
        **kwargs):
        """Submit simulation jobs"""

        script = path.join(path.dirname(simCRpropa.__file__), 'scripts/run_crpropa_em_cascade.py')

        if not path.isfile(script):
            raise IOError("Script {0:s} not found!".format(script))

        for ib, b in enumerate(self._bList):
            for il, l in enumerate(self._turbScaleList):
                njobs = int(self.Simulation['Nparticle']) / int(self.Simulation['Nbatch'])
                self.Bfield['B'] = b
                self.Bfield['maxTurbScale'] = l
                self.setOutput(0, idB = ib, idL = il)

                outfile = path.join(self.FileIO['outdir'],self.OutName.split('_')[0] + '*.hdf5')
                missing = utils.missing_files(outfile,njobs, split = '.hdf5')
                self.config['Simulation']['n_cpu'] = kwargs['n']


                if len(missing) < njobs:
                    logging.debug('here {0}'.format(njobs))
                    njobs = missing
                    logging.info('there are {0:n} files missing in {1:s}'.format(len(missing),
                    outfile ))

                if len(missing) and not force_combine:
                    self.config['configname'] = 'r'
                    kwargs['logdir'] = path.join(self.FileIO['outdir'],'log/')
                    kwargs['tmpdir'] = path.join(self.FileIO['outdir'],'tmp/')
                    kwargs['jname'] = 'b{0:.0f}l{1:.0f}'.format(np.log10(b),np.log10(l))
                    lsf.submit_lsf(script,
                        self.config,'',njobs, 
                        **kwargs)
                else:
                    if len(missing) and force_combine:
                        logging.info("There are files missing but combining anyways.")
                    else:
                        logging.info("All files present.")

                    ffdat = glob(path.join(path.dirname(outfile),path.basename(outfile).split('.hdf5')[0] + '.dat'))
                    if len(ffdat):
                        logging.info("Deleting *.dat files.")
                        for f in ffdat:
                            utils.rm(f)


                    collect.combine_output(outfile, overwrite = overwrite)
        return

@lsf.setLsf
def main(**kwargs):
    usage = "usage: %(prog)s"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('--conf', required = True)
    parser.add_argument('--dry', default = 0, type = int)
    parser.add_argument('--time', default = '09:59',help='Max time for lsf cluster job')
    parser.add_argument('--n', default = 8,help='number of reserved cores', type=int)
    parser.add_argument('--span', default = 'span[ptile=8]',help='spanning of jobs')
    parser.add_argument('--concurrent', default = 0,help='number of max simultaneous jobs', type=int)
    parser.add_argument('--sleep', default = 10,help='seconds to sleep between job submissions', type=int)
    parser.add_argument('--overwrite', default = 0,help='overwrite existing combined files', type=int)
    parser.add_argument('--force_combine', default = 0,help='force the combination of files', type=int)
    args = parser.parse_args()
    kwargs['dry'] = args.dry
    kwargs['time'] = args.time
    kwargs['concurrent'] = args.concurrent
    kwargs['sleep'] = args.sleep
    kwargs['n'] = args.n
    kwargs['span'] = args.span
    
    utils.init_logging('DEBUG', color = True)
    config = yaml.load(open(args.conf))
    sim = SimCRPropa(**config)
    sim.run(overwrite = bool(args.overwrite),
        force_combine = bool(args.force_combine), **kwargs)
    return

if __name__ == '__main__':
    main()