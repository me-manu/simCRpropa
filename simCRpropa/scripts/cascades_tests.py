import numpy as np
import os
import argparse
from fermiAnalysis.batchfarm import utils

from crpropa import *

#
def process(fn):
    """
    ABOUT
    -----
    Process a given simulation file using simple cuts.

    INPUT
    -----
      fn: name of the file (CRPropa simulation with 'Event1D' output including weight column)

    OUTPUT
    ------
    """
    f = open(fn,'rt').readlines()[:-1]
    data = np.genfromtxt(f, dtype = np.float64, comments = '#')
    idx = np.where(data[:, 1] == 22)
    e = data[idx, 2]
    w = data[idx, 5]

    bins = np.logspace(9, 15, 61)
    h, _ = np.histogram(e, bins=bins, weights = w)
    
    return h, bins, data

# ____________________________________________________________________________________________
#
def simulate(nEvents, z, e, ebl, runSim = True, out_dir=None, thinning=1.):

    d = redshift2ComovingDistance(z)

    if ebl == 'G12':
        EBL = IRB_Gilmore12
    elif ebl == 'F08':
        EBL = IRB_Franceschini08

    outputName = 'sim/cascades/sim-cascades-z_%3.2f-E_%2.1eeV-%s.txt' % (z, e / eV, ebl)
    if out_dir is not None:
        outputName = os.path.join(out_dir, outputName)
    if not os.path.exists(os.path.dirname(outputName)):
        os.makedirs(os.path.dirname(outputName))

    # single source
    source = Source()
    source.add(SourcePosition(Vector3d(d, 0, 0)))
    source.add(SourceDirection(Vector3d(-1, 0, 0)))
    source.add(SourceParticleType(22))
    source.add(SourceEnergy(e))
    source.add(SourceRedshift1D())

    # output
    output = TextOutput(outputName, Output.Event1D)
    output.setEnergyScale(eV)
    output.set(output.WeightColumn, True)

    # observer
    observer = Observer()
    observer.add(ObserverPoint())
    observer.onDetection(output)

    # module setup
    m = ModuleList()
    m.add(SimplePropagation(1e-1 * kpc, 10 * Mpc))
    m.add(Redshift())
    m.add(EMPairProduction(CMB, True, thinning))
    m.add(EMPairProduction(EBL, True, thinning))
    m.add(EMInverseComptonScattering(CMB, True, thinning))
    m.add(EMInverseComptonScattering(EBL, True, thinning))
    m.add(MaximumTrajectoryLength(4000 * Mpc))
    m.add(MinimumEnergy(1e9 * eV))
    m.add(observer)

    # run simulation
    m.setShowProgress(True)
    m.run(source, nEvents, True)

    h, bins, data = process(outputName)
    return h, bins, data

# ____________________________________________________________________________________________
#
def plot(graphs, plotName, title = ''):
    """
    """

# ____________________________________________________________________________________________
#
if __name__ == '__main__':
    usage = "usage: %(prog)s --conf config.yaml"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-o','--out_dir', required = True)
    parser.add_argument('-t','--thinning', type=float, default=1.)
    parser.add_argument('-n','--nevents', type=int, default=1000)
    parser.add_argument('-e','--energy', type=float, default=1e14)
    args = parser.parse_args()
    utils.init_logging('DEBUG', color = False)
    

    ## run the simulations
    # simulate(10000, 0.02, 1e13 * eV, 'F08')
#    simulate(10000, 0.20, 1e13 * eV, 'F08')
#    simulate(10000, 0.05, 1e14 * eV, 'F08')
    h, bins, data = simulate(args.nevents, 0.14, args.energy * eV, 'F08', out_dir=args.out_dir, thinning=args.thinning)
