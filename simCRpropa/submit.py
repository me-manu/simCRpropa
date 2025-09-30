import os
import sys
import logging
import time
import resource
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
        tmpdir = os.path.join(tmpdir, '{0:s}.XXXXXX'.format(os.environ["LSB_JOBID"]))
        p = Popen(["mktemp", "-d", tmpdir], stdout=PIPE)  # make a temporary directory and get its name
        time.sleep(10.)
        out, err = p.communicate()

        logging.info('out: {0}'.format(out))
        logging.info('err: {0}'.format(err))

        tmpdir = os.path.join('/scratch', out.decode('ascii').split()[0])
        time.sleep(10.)
    except ValueError:
        job_id = local_id
        tmpdir = os.path.join(os.environ["PWD"], 'tmp/')
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
        tmpdir = os.path.join(tmpdir, '{0:s}.XXXXXX'.format(os.environ["SLURM_JOB_ID"]))
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

    return tmpdir, job_id


def limit_memory(maxsize):
    """maxsize in Gbyte"""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if maxsize * 1e6 > virtual_memory().total:
        maxsize = int(virtual_memory().total / 1e6)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize * int(1e6), hard))
    logging.info("limited memory to {0:d} Mbytes".format(maxsize))
    return
