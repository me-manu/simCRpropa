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
from simCRpropa.collect import readCRPropaOutput,convertOutput2Hdf5
from simCRpropa import collect as col
from fermiAnalysis.batchfarm import utils, lsf, sdf
from crpropa import *
from psutil import virtual_memory
from subprocess import call, check_call, Popen, PIPE, check_output, CalledProcessError

def init_lsf(local_id=0):
    """
    Init lsf cluster jobs: set up tmpdir on cluster scratch, determine job_id and set pfiles to tmpdir on scratch

    kwargs
    ------
    local_id:        int, if not on lsf, return this value as job_id

    Returns
    -------
    tuple with tmpdir on cluster scratch and lsf job id
    """

    try:
        list(os.environ.keys()).index("LSB_JOBNAME")
        job_id = int(os.environ["LSB_JOBINDEX"])
        tmpdir = os.path.join('/scratch/{0:s}.{1:s}/'.format(os.environ['USER'],os.environ["LSB_JOBID"]))
        if not os.path.exists(tmpdir):
            tmpdir = os.mkdir(tmpdir)
        logging.info('os.listdir: {0}'.format(os.listdir(tmpdir)))

        time.sleep(10.)
        tmpdir = os.path.join(tmpdir,'{0:s}.XXXXXX'.format(os.environ["LSB_JOBID"]))
        p = Popen(["mktemp", "-d", tmpdir], stdout=PIPE)  # make a temporary directory and get its name
        time.sleep(10.)
        out, err = p.communicate()

        logging.info('out: {0}'.format(out))
        logging.info('err: {0}'.format(err))

        tmpdir = os.path.join('/scratch',out.decode('ascii').split()[0])
        time.sleep(10.)
    except ValueError:
        job_id = local_id
        tmpdir = os.path.join(os.environ["PWD"],'tmp/')
        if not os.path.exists(tmpdir):
            tmpdir = os.mkdir(tmpdir)

    logging.info('tmpdir is {0:s}.'.format(tmpdir))

    if not os.path.exists(tmpdir):
        logging.error('Tmpdir does not exist: {0}. Exit 14'.format(tmpdir))
        sys.exit(14)

    return tmpdir, job_id

def init_sdf(local_id=0):
    """
    Init sdf cluster jobs: set up tmpdir on cluster scratch, determine job_id and set pfiles to tmpdir on scratch

    kwargs
    ------
    local_id: int
        if not on lsf, return this value as job_id

    Returns
    -------
    tuple with tmpdir on cluster scratch and lsf job id
    """

    try:
        list(os.environ.keys()).index("SLURM_JOB_NAME")
        job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tmpdir = os.environ["LSCRATCH"]
        logging.info('os.listdir: {0}'.format(os.listdir(tmpdir)))

        time.sleep(1.)
        tmpdir = os.path.join(tmpdir,'{0:s}.XXXXXX'.format(os.environ["SLURM_JOB_ID"]))
        p = Popen(["mktemp", "-d", tmpdir], stdout=PIPE)  # make a temporary directory and get its name
        time.sleep(1.)
        out, err = p.communicate()

        logging.info('out: {0}'.format(out))
        logging.info('err: {0}'.format(err))

        tmpdir = os.path.join(os.environ["LSCRATCH"], out.decode('ascii').split()[0])
        time.sleep(1.)
    except ValueError as e:
        logging.error("Received error {0}".format(e))
        job_id = local_id
        tmpdir = os.path.join(os.environ["PWD"],'tmp/')
        if not os.path.exists(tmpdir):
            tmpdir = os.mkdir(tmpdir)

    logging.info('tmpdir is {0:s}.'.format(tmpdir))

    if not os.path.exists(tmpdir):
        logging.error('Tmpdir does not exist: {0}. Exit 14'.format(tmpdir))
        sys.exit(14)

    return tmpdir,job_id

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

    logging.info("Host name: {0}".format(socket.gethostname()))

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
        if not i:
            logging.info(sim.source)
            logging.info(sim.m)
            logging.info(sim.observer)

        if not sim.Source['useSpectrum']:
            sim.Source['Energy'] = sim.EeV[i]
            logging.info("======= Bin {0:n} / {1:n}, Energy : {2:3e} eV ========".format(
                i + 1, sim.nbins, sim.EeV[i]))
        t0 = time.time()

        sim._create_source()
        # run simulation
        if not job_id or args.show_progress:
            sim.m.setShowProgress(True)
        else:
            sim.m.setShowProgress(False)
        logging.info("Running simulation for {1:n} particle(s), saving output to {0:s}".format(sim.outputfile,
                sim.weights[i]))

        # void run(SourceInterface *source, size_t count, bool recursive = true, bool secondariesFirst = false)
        sim.m.run(sim.source,  int(sim.weights[i]), True, True)
        sim.output.close()
        logging.info("Simulating bin {0:n} / {1:n} took {2:.1f} s".format(i + 1, sim.nbins, time.time() - t0))

    utils.sleep(1.)

    outputfile = str(deepcopy(sim.outputfile))
    outdir = deepcopy(sim.FileIO['outdir'])
    useSpectrum = deepcopy(sim.Source['useSpectrum'])
    weights = deepcopy(sim.weights)
    outtype = deepcopy(sim.Simulation['outputtype'])
    del sim # free memorY

    # read the output
    if outtype == 'ascii':
        names, units, data = readCRPropaOutput(outputfile)
        hfile = outputfile.split(".dat")[0] + ".hdf5"

        col.convertOutput2Hdf5(names, units, data, weights, hfile, config,
                  pvec_id = ['','0'],
                  xvec_id = ['','0'],
                  useSpectrum = useSpectrum)

        #utils.zipfiles(sim.outputfile,sim.outputfile + '.gz', nodir = True)
        utils.copy2scratch(hfile, outdir)

    else:
        utils.copy2scratch(outputfile, outdir)

    utils.sleep(1.)
