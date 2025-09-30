import os
import logging
import numpy as np
import yaml
import argparse
import resource
import socket
import os
import time
from os import path
from glob import glob
from copy import copy, deepcopy
from simCRpropa.sim_crpropa import SimCRPropa
from simCRpropa.collect import readCRPropaOutput, convertOutput2Hdf5
from simCRpropa import collect as col
from fermiAnalysis.batchfarm import utils, lsf, sdf
from crpropa import *
from psutil import virtual_memory
from subprocess import call, check_call, Popen, PIPE, check_output, CalledProcessError
from simCRpropa.submit import init_lsf, init_sdf, limit_memory


def RunSim(N, OutputName, B, show_progress=True):
    """
    Run the simulation with fixed parameters. Function adapted from Paolo Da Vela

    Parameters
    ----------
    N: int
        number of injected particles

    OutputName: str
        path to output file

    B: float
        RMS of magnetic field strength in Gauss

    show_progress: bool
        If true, show progress bar
    """

    # fix cosmology
    h = 0.7
    Om = 0.3
    z = 0.140
    setCosmologyParameters(h, Om)
    D = redshift2ComovingDistance(z)

    # set resolution
    minStep = 1e-5 * kpc
    maxStep = 10. * Mpc
    tol = 1e-11
    thinning = 0.

    # spectral index
    index = -1.5

    # half aperture of source
    jetAngle = 5.0

    # injected energies and min Rigidity
    rigidity = 50. * 1e9 * eV
    Emin = 1e11 * eV
    Emax = 5e13 * eV

    # *** Setting up the B-field ***
    gridPoints = 100
    gridSize = 50. * Mpc
    gridSpacing = gridSize / gridPoints
    gridprops = GridProperties(Vector3d(0), gridPoints, gridSpacing)
    #randomSeed = random.randint(1, 1e6)
    randomSeed = 42  # use a fixed seed so that all simulations use same B
    minScale = 2 * gridSpacing
    maxScale = 25. * Mpc
    turbSpectrum = SimpleTurbulenceSpectrum(B * gauss, minScale, maxScale, 5. / 3)
    BField = SimpleGridTurbulence(turbSpectrum, gridprops, randomSeed)

    # print some properties of our field
    print('Lc = {:.1f} Mpc'.format(BField.getCorrelationLength() / Mpc))  # correlation length
    print('sqrt(<B^2>) = {:e} nG'.format(BField.getBrms() / nG))  # RMS
    print('<|B|> = {:.12f} nG'.format(BField.getMeanFieldStrength() / nG))  # mean
    print('B(10 Mpc, 0, 0) =', BField.getField(Vector3d(10, 0, 0) * Mpc) / nG, 'nG')

    # *** Setting up the source ***
    source = Source()
    source.add(SourcePosition(0))
    source.add(SourceParticleType(22))
    # source.add(SourceEnergy(E)) #monochromatic photon
    source.add(SourcePowerLawSpectrum(Emin, Emax, index))  # powerlaw

    # spectrumStr = "E^-2.8*exp(-E/(30*TeV))"
    # spectrumStr = "E^(-2.38-0.1*log10(E))"
    # spectrumStr = "log10(E)"
    # print(Emin, "  ", Emax)
    # genericSourceComposition = SourceGenericComposition(Emin, Emax, spectrumStr)
    # genericSourceComposition.add(22,1)
    # source.add(genericSourceComposition)
    source.add(SourceEmissionCone(Vector3d(1, 0, 0), np.radians(jetAngle)))

    # *** Setting up the observer ***
    output = TextOutput(OutputName, Output.Everything)
    output.setLengthScale(kpc)
    obsPos = Vector3d(0, 0, 0)
    observer = Observer()
    observer.add(ObserverSurface(Sphere(obsPos, D)))
    observer.add(ObserverElectronVeto())
    observer.onDetection(output)

    simulation = ModuleList()
    simulation.add(PropagationCK(BField, tol, minStep, maxStep))
    simulation.add(FutureRedshift())
    simulation.add(EMInverseComptonScattering(CMB(), True, thinning))
    simulation.add(EMPairProduction(CMB(), True, thinning))
    simulation.add(EMPairProduction(IRB_Franceschini08(), True, thinning))
    simulation.add(SynchrotronRadiation(BField))
    simulation.add(MinimumRigidity(rigidity))
    simulation.add(MinimumEnergy(1e9 * eV))
    simulation.add(observer)
    simulation.showModules()
    simulation.setShowProgress(show_progress)
    simulation.run(source, N, True)


if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c', '--conf', required = True)
    parser.add_argument('-i', required=False, default=0, 
                        help='Set local or scratch calculation', type=int)
    parser.add_argument('-l', required=False, default=0, 
                        help='If > 0, limit memory to this size in mega bytes', type=int)
    parser.add_argument('-b', '--batch-farm-name', default='sdf', choices=['sdf', 'lsf'],
                        help='name of batch farm being used')
    parser.add_argument('--show-progress', action="store_true",
                        help='show the progress bar')
    args = parser.parse_args()
    utils.init_logging('DEBUG', color = False)

    if args.l > 0:
        limit_memory(args.l)

    with open(args.conf) as f:
        config = yaml.safe_load(f)

    if args.batch_farm_name == 'sdf':
        tmpdir, job_id = init_sdf(local_id=args.i)
    elif args.batch_farm_name == 'lsf':
        tmpdir, job_id = init_lsf(local_id=args.i)
    else:
        raise ValueError(f"{args.batch_farm_name} not understood")

    logging.info("Host name: {0}".format(socket.gethostname()))

    if not job_id:
        job_id = args.i
    logging.info('tmpdir: {0:s}, job_id: {1:d}'.format(tmpdir, job_id))
    os.chdir(tmpdir)    # go to tmp directory
    logging.info('Entering directory {0:s}'.format(tmpdir))
    logging.info('PWD is {0:s}'.format(os.environ["PWD"]))

    sim = SimCRPropa(**config)
# limit number of used threads -- does not really work... 
    os.environ['OMP_NUM_THREADS'] = str(sim.Simulation['cpu_n'])

    sim.setOutput(job_id)
    sim.outputfile = str(path.join(tmpdir, path.basename(sim.outputfile)))
    logging.info("writing output file to : {0:s}".format(sim.outputfile))
    logging.info("and will copy it to : {0:s}".format(sim.FileIO['outdir']))
    sim.setup()

    config['Source']['LightTravelDistance'] = redshift2LightTravelDistance(config['Source']['z'])
    config['Source']['LuminosityDistance'] = redshift2LuminosityDistance(config['Source']['z'])
    config['Source']['ComovingDistance'] = redshift2ComovingDistance(config['Source']['z'])

    t00 = time.time()

    RunSim(sim.Simulation['Nbatch'], sim.outputfile, sim.Bfield['B'],
           show_progress=True if (not job_id or args.show_progress) else False)

    logging.info("Total simulation took {0:.1f} s".format(time.time() - t00))

    utils.sleep(1.)

    outputfile = str(deepcopy(sim.outputfile))
    outdir = deepcopy(sim.FileIO['outdir'])
    useSpectrum = False
    outtype = 'ascii'
    del sim  # free memory

    # read the output
    if outtype == 'ascii':
        names, units, data = readCRPropaOutput(outputfile)
        hfile = outputfile.split(".dat")[0] + ".hdf5"

        col.convertOutput2Hdf5(names, units, data, weights, hfile, config,
                               pvec_id=['', '0'],
                               xvec_id=['', '0'],
                               useSpectrum=useSpectrum)

        #utils.zipfiles(sim.outputfile,sim.outputfile + '.gz', nodir = True)
        utils.copy2scratch(hfile, outdir)

    # also copy the raw text files
    utils.copy2scratch(outputfile, outdir)

    utils.sleep(1.)
