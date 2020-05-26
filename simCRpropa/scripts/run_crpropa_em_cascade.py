import os
import logging
import numpy as np
import yaml
import argparse
import resource
from time import time
from os import path
from glob import glob
from copy import copy, deepcopy
from simCRpropa.sim_crpropa import SimCRPropa
from simCRpropa.collect import readCRPropaOutput,convertOutput2Hdf5
from simCRpropa import collect as col
from fermiAnalysis.batchfarm import utils,lsf
from crpropa import *
from psutil import virtual_memory

def limit_memory(maxsize):
    "maxsize in Gbyte"
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if maxsize * 1e6 > virtual_memory().total:
        maxsize = int(virtual_memory().total / 1e6)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize * int(1e6), hard))
    logging.info("limited memory to {0:n} Mbytes".format(maxsize))
    return


if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-c','--conf', required = True)
    parser.add_argument('-i', required=False, default = 0, 
                        help='Set local or scratch calculation', type=int)
    parser.add_argument('-l', required=False, default = 0, 
                        help='If > 0, limit memory to this size in mega bytes', type=int)
    args = parser.parse_args()
    utils.init_logging('DEBUG', color = False)

    if args.l > 0:
        limit_memory(args.l)

    with open(args.conf) as f:
        config = yaml.safe_load(f)

    tmpdir, job_id = lsf.init_lsf()
    if not job_id:
        job_id = args.i
    logging.info('tmpdir: {0:s}, job_id: {1:n}'.format(tmpdir,job_id))
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

#    weights = sim.Simulation['Nbatch'] * \
#        np.ones(nbins, dtype = np.int) # weight with optical depth?
#    if sim.config['Observer']['obsSmallSphere']:
#        # weight for Bfield
#        if sim.config['Bfield']['B'] > 1e-18:
#            weights *= (1. + 0.1 * (np.log10(sim.config['Bfield']['B']) + 18.)**2.)
#        # weight for jet opening angle 
#        if sim.config['Source']['th_jet'] > 1.:
#            weights *= (1. + 0.1 *sim.config['Source']['th_jet']**2.)
#        if sim.config['Observer']['obsAngle'] > 0.:
#            weights *= (1. + 0.1 * (sim.config['Observer']['obsAngle'] + 1.))
#    logging.info('weights: {0}'.format(weights))

    # add distances to config
    config['Source']['LightTravelDistance'] = redshift2LightTravelDistance(config['Source']['z'])
    config['Source']['LuminosityDistance'] = redshift2LuminosityDistance(config['Source']['z'])
    config['Source']['ComovingDistance'] = redshift2ComovingDistance(config['Source']['z'])

    for i in range(sim.nbins):

        if not sim.Source['useSpectrum']:
            sim.Source['Energy'] = sim.EeV[i]
            logging.info("======= Bin {0:n} / {1:n}, Energy : {2:3e} eV ========".format(
                i + 1, sim.nbins, sim.EeV[i]))

        sim._create_source()
        # run simulation
        if not job_id:
            sim.m.setShowProgress(True)
        else:
            sim.m.setShowProgress(False)
        if not i:
            sim.m.showModules()
            logging.info(sim.source)
            logging.info(sim.m)
            logging.info(sim.observer)
        logging.info("Running simulation for {1:n} particles, saving output to {0:s}".format(sim.outputfile,
                sim.weights[i]))
        sim.m.run(sim.source,  sim.weights[i], True)
        sim.output.close()

    utils.sleep(1.)

    outputfile = str(deepcopy(sim.outputfile))
    outdir = deepcopy(sim.FileIO['outdir'])
    useSpectrum = deepcopy(sim.Source['useSpectrum'])
    weights = deepcopy(sim.weights)
    del sim # free memorY
    # read the output
    names, units, data = readCRPropaOutput(outputfile)
    hfile = outputfile.split(".dat")[0] + ".hdf5"

    col.convertOutput2Hdf5(names, units, data, weights, hfile, config,
              pvec_id = ['','0'],
              xvec_id = ['','0'],
              useSpectrum = useSpectrum)

    #utils.zipfiles(sim.outputfile,sim.outputfile + '.gz', nodir = True)
    utils.copy2scratch(hfile, outdir)
    utils.sleep(1.)
