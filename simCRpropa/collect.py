"""
Functions and classes to collect results 
from CRPropa simulations
"""

import logging
import yaml
import numpy as np
import h5py
try:
    from crpropa import *
except:
    pass
from os import path
try:
    from fermiAnalysis.batchfarm import utils, lsf
except:
    pass
from glob import glob
from scipy.integrate import simps
from collections import OrderedDict
from collections.abc import Iterable


def combine_output(outfile, overwrite = False):
    """
    Combine the output hdf5 files
    into one file

    Parameters
    ----------
    outfile: str
    globable string to outfile paths
    """
    if path.isfile(path.join(path.dirname(outfile),'combined.hdf5')) and not overwrite:
        logging.info("Combined file present and overwrite set to False.")
        return

    ff = glob(outfile)
    # get the data and concatenate it
    ff = sorted(ff, key = lambda f : int(path.basename(f).split('.hdf5')[0][-5:]))
    if path.exists(path.join(path.dirname(outfile),'combined.hdf5')) and overwrite:
        utils.rm(path.join(path.dirname(outfile),'combined.hdf5'))

    combined = h5py.File(path.join(path.dirname(outfile),'combined.hdf5'), 'w')

    where_to_start_appending = {}
    final_rows = {}

    logging.info("Combining hdf5 files...")

    ifile = 0
    skipped_files = []
    for i,fi in enumerate(ff):
        logging.info("Working on {0:s} ...".format(fi))
        f = h5py.File(fi, "r+")
        conf = yaml.safe_load(f['simEM'].attrs['config'])
        try:
            f['simEM'].keys()
        except (RuntimeError, ValueError):
            logging.error("There was a problem with {0:s}\nContinuing with next file".format(fi))
            continue

        if isinstance(conf['Source']['Emin'], float):
            esteps = conf['Source']['Esteps'] - 1
        #elif isinstance(conf['Source']['Emin'], list) or isinstance(conf['Source']['Emin'], tuple) or \
        #    isinstance(conf['Source']['Emin'], np.ndarray):
        elif isinstance(conf['Source']['Emin'], Iterable):
            esteps = len(conf['Source']['Emin'])


        if not conf['Source']['useSpectrum']:
            for name in f['simEM']:                                                                        
                # check if number of bins is correct
                if not 'weights' in f['simEM'][name].keys():
                    if not len(f['simEM'][name].keys()) == esteps:
                        logging.error("Energy bins missing in {0:s}".format(fi))
                        skipped = True
                        skipped_files.append(i)
                        break
                    else:
                        skipped = False

                for Eb in f['simEM'][name]:                                                                
                    k = 'simEM/' + name + '/' + Eb  
                    if not ifile:
                        final_rows[k] = 0

                    if ifile and (k == 'simEM/intspec/Ecen' or k == 'simEM/intspec/weights'):
                        continue

                    if len(f[k].shape) == 1:
                        final_rows[k] = final_rows[k] + f[k].shape[0]
                    elif len(f[k].shape) == 2: # position vectors
                        final_rows[k] = final_rows[k] + f[k].shape[1]
            f.close()
            if not skipped:
                ifile += 1
        else:
            for name in f['simEM']:                                                                        
                k = 'simEM/' + name
                if not ifile:
                    final_rows[k] = 0

                if (name  == 'intspec'):
                    continue

                if len(f[k].shape) == 1:
                    final_rows[k] = final_rows[k] + f[k].shape[0]
                elif len(f[k].shape) == 2: # position vectors
                        final_rows[k] = final_rows[k] + f[k].shape[1]
            f.close()
            ifile += 1


    ifile = 0
    for i,fi in enumerate(ff):
        if i in skipped_files:
            utils.rm(fi)
            continue

        f = h5py.File(fi, "r+")
        try:
            f['simEM'].keys()
        except RuntimeError:
            logging.error("There was a problem with {0:s}\nContinuing with next file".format(fi))
            continue
        for name in f['simEM']:                                                                        
            # we have simulated a spectrum
            if conf['Source']['useSpectrum']:
                k = 'simEM/' + name  
                if name == 'intspec':
                    continue

                if ifile == 0:
                    where_to_start_appending[k] = 0
                    #first file; create the dummy dataset with no max shape
                if len(f[k].shape) == 2 or (f[k].shape[0] == 0 and (k.find('X') >= 0 or k.find('P') >= 0)):

                    if not k in combined:
                        combined.create_dataset(k, 
                            (3,final_rows[k]),
                            dtype = f[k].dtype,
                            compression="gzip")  

                elif len(f[k].shape) == 1:

                    if not k in combined:
                        combined.create_dataset(k,
                            (final_rows[k],),
                            dtype = f[k].dtype,
                            compression="gzip")
                if len(f[k].shape) == 1:
                    try:
                        combined[k][where_to_start_appending[k]:where_to_start_appending[k] + f[k].shape[0]] = \
                            f[k]
                        where_to_start_appending[k] += f[k].shape[0]
                    except IOError:
                        logging.error("There was a problem (IOError) with {0:s}\nContinuing with next file".format(fi))
                        continue
                elif len(f[k].shape) == 2:
                    combined[k][:,where_to_start_appending[k]:where_to_start_appending[k] + f[k].shape[1]] = f[k]
                    where_to_start_appending[k] += f[k].shape[1]

            # we have simulated individual bins
            else:
                for Eb in f['simEM'][name]:                                                                
                    k = 'simEM/' + name + '/' + Eb  
                    if ifile == 0:
                        where_to_start_appending[k] = 0
                        #first file; create the dummy dataset with no max shape
                    if len(f[k].shape) == 2 or (f[k].shape[0] == 0 and (k.find('X') >= 0 or k.find('P') >= 0)):

                        if not k in combined:
                            combined.create_dataset(k, 
                                (3,final_rows[k]),
                                dtype = f[k].dtype,
                                compression="gzip")  

                    elif len(f[k].shape) == 1:

                        if not k in combined:
                            combined.create_dataset(k,
                                (final_rows[k],),
                                dtype = f[k].dtype,
                                compression="gzip")
                    if len(f[k].shape) == 1:
                        if i and (k == 'simEM/intspec/Ecen' or k == 'simEM/intspec/weights'):
                            continue
                        try:
                            combined[k][where_to_start_appending[k]:where_to_start_appending[k] + f[k].shape[0]] = \
                                f[k][()] * len(ff) if k == 'simEM/intspec/weights' else f[k]
                            where_to_start_appending[k] += f[k].shape[0]
                        except IOError:
                            logging.error("There was a problem (IOError) with {0:s}\nContinuing with next file".format(fi))
                            continue
                    elif len(f[k].shape) == 2:
                        combined[k][:,where_to_start_appending[k]:where_to_start_appending[k] + f[k].shape[1]] = f[k]
                        where_to_start_appending[k] += f[k].shape[1]

        # copy config from last used file to combined file
        if i == len(ff) - 1:
            combined['simEM'].attrs['config'] = f['simEM'].attrs['config']
        f.close()
        ifile += 1
    combined.close()
    logging.info("Done.")

    return


def convertOutput2Hdf5(names, units, data, weights, hfile,
              config,
              pvec_id = [''],
              xvec_id = [''],
              useSpectrum = False):
    """
    Convert CRPropa Output to an HDF5 file

    Parameters
    ----------
    names: list
        list of strings with column names

    units: dict 
        dictionary with column units

    data: `~numpy.ndarray`
        CRPropa output data

    weights: `~numpy.ndarray`
        weights for the simulation / number of injected particles
    
    hfile: str
        path to output file

    config: dict  
        config dict

    {options}
    pvec_id: list 
        list of strings that will be appended to momentum  
        columns which will be turned into vector columns.

    xvec_id: list 
        list of strings that will be appended to position
        columns which will be turned into vector columns.

    useSpectrum: bool
        if False, assume bin-by-bin simulation (default: False)
    """
    # stack vectors together:
    idx = ['X','Y','Z']
    idp = ['Px','Py','Pz']
    pos_vectors = []
    mom_vectors = []
    for v in xvec_id:
        pos_vectors.append( np.vstack(
            [data[:,names.index(i + v)] for i in idx]))
    for v in pvec_id:
        mom_vectors.append( np.vstack(
            [data[:,names.index(i[:1] + v + i[1:])] for i in idp]))
    pos_vectors = np.array(pos_vectors) # shape is (len(vec_id),3,data.shape[0])
    mom_vectors = np.array(mom_vectors) # shape is (len(vec_id),3,data.shape[0])

    # init hdf5 file
    h = h5py.File(hfile, 'a')
    if "simEM" in h.keys(): # overwrite group if already exists
        del h["simEM"]
    grp = h.create_group("simEM")
    grp.attrs["config"] = yaml.safe_dump(config,default_flow_style=False)
    for i,n in enumerate(names):
        if n in ['Y','Z', 'Y0','Z0', 'Y1','Z1']: continue
        if n in ['Py','Pz', 'P0y','P0z', 'P1y','P1z']: continue


        if n in ['X' + v for v in xvec_id]:
            d = pos_vectors[['X' + v for v in xvec_id].index(n),...]
        elif n in ['P' + v + 'x' for v in pvec_id]:
            d = mom_vectors[['P' + v + 'x' for v in pvec_id].index(n),...]
        else:
            if n.find('ID') >= 0:
                d = data[:,i].astype(int)
            else:
                d = data[:,i]

        if n.find('ID') >= 0:
            dtype = 'i8'
        else:
            dtype ='f8'

        if not useSpectrum:
            if isinstance(config['Source']['Emin'], float):
                EeVbins = np.logspace(np.log10(config['Source']['Emin']),
                    np.log10(config['Source']['Emax']), config['Source']['Esteps'])

            elif isinstance(config['Source']['Emin'], Iterable):
                EeVbins = np.append(config['Source']['Emin'], config['Source']['Emax'][-1])

            e0i = names.index('E0') # index of injected energy
            Ecen = np.unique(data[:,e0i])
            grp.create_group(n)
            # for bin-by-bin simulations:
            # create a data set for each bin of injected energy
            # these will have different lenghts, since 
            # higher energy particles will cause more cascade photons
            for j,E in enumerate(EeVbins[:-1]):
            #for j,E in enumerate(Ecen):
                #ide = np.where(data[:,e0i] == E)
                ide = np.where((data[:,e0i] >= E) & \
                            (data[:,e0i] < EeVbins[j+1]))

                if len(ide[0]):
                    grp[n].create_dataset("Ebin{0:03n}".format(j), dtype = dtype,
                                        data = d[...,ide[0]], compression="gzip")
                # no such injected energy in output file
                # create an empty data set
                else:
                    logging.warning("No particles with injected energy {0:.3e} found!".format(E))
                    grp[n].create_dataset("Ebin{0:03n}".format(j), dtype = dtype,
                                        data = [], compression="gzip")

                grp["{0:s}/Ebin{1:03n}".format(n,j)].attrs['unit'] = units[n]
            logging.info("{0} Ebin{1:03} {2}".format(n,j,grp["{0:s}/Ebin{1:03n}".format(n,j)]))
        else:
            grp.create_dataset(n, dtype = dtype,
                data = d, compression="gzip")
            grp[n].attrs['unit'] = units[n]

    intspec = grp.create_group("intspec")
    if not useSpectrum:
        intspec.create_dataset("Ecen", data = Ecen,
            compression="gzip" ) 
        intspec["Ecen"].attrs['unit'] = 'eV'

        intspec.create_dataset("weights", data = weights,
            compression = "gzip") 
    h.close()
    return 

def readCRPropaOutput(filename):
    """Read the header of a CRPROPA outputfile"""
    units = OrderedDict()
    data = np.loadtxt(filename)

    with open(filename) as h:
        for i,l in enumerate(h.readlines()):
            if not l[0] == '#': continue
            if not len(l.split()): continue
            if not i:
                names = l.lstrip('#').split() # get the name columns
            # extract units
            if l.split()[-1].find(']') >= 0:
                par = l.lstrip('#').split()[0][0]
                if float(l.split()[-2].strip('[')) == 1.:
                    units[par] = '{0:s}'.format(l.split()[-1].strip(']'))
                else:
                    units[par] = '{0:s} {1:s}'.format(l.split()[-2].strip('['),l.split()[-1].strip(']'))

        if 'X' in units.keys(): 
            for par in ['Y','Z']:
                units[par] = units['X']
        for i in range(2):
            if 'X{0:n}'.format(i) in names:
                for par in ['X{0:n}'.format(i), 'Y{0:n}'.format(i),'Z{0:n}'.format(i)]:
                    units[par] = units['X']
            if 'E{0:n}'.format(i) in names:
                units['E{0:n}'.format(i)] = units['E']

        formats = []
        for i,n in enumerate(names):
            if n.find('ID') >= 0:
                formats.append('I')
            else:
                formats.append('D')
            if not n in units.keys():
                units[n] = ''

    return names, units, data


class EMHist(object):
    """
    Class for creation and manipulation 
    of N-dim histogram with CRPropa output for electromagnetic 
    cascade and histogram for intrinsic spectrum
    """
    def __init__(self, Einj, Eobs, dt,
                   # dtheta,
                    phi, theta, idobs, id1, bins,injected ,
                    idinj  = 22 , iddetection= 22, steps = 10,
                    degperpix = 0.04,
                    tbinslog = True,
                    config = None):
        """
        Create the histogram

        Parameters
        ----------
        Einj: `~numpy.ndarray`
            Array with injected energies in eV
        Eobs: `~numpy.ndarray`
            Array with observed energies in eV
        dt: `~numpy.ndarray`
            Array with time delays in years
        phi: `~numpy.ndarray`
            Array with phi coordinates (l or DEC) in degrees
        theta: `~numpy.ndarray`
            Array with theta coordinates (pi / 2. - b or RA) in degrees
        idobs: `~numpy.ndarray`
            Array with observed particle IDs
        id1: `~numpy.ndarray`
            Array with particle IDs of parent particles
        bins: list
            list with bin edges for all axis
        injected: `~numpy.ndarray`
            (2xN) dim array with the central energies and counts of 
            injected particles

        {options}
       
        idinj: int
            particle ID of injected particles
            (used to determine if particle is produced in cascade, default: 22)
        iddetection: int
            particle ID for detected particle
            (veto against other particles, default: 22)
        steps: int
            steps for integration of intrinsic spectrum (default: 10)
        degperpix: float
            assumed pixelization of skymaps
        tbinslog: bool
            if true, time bins considered to be log spaced
        """
        self.__degperpix = degperpix
        self._injected = injected
        # cascade condition
        self._mc = (idobs == iddetection) & (id1 != idinj)
        # injected spectrum condition
        self._mi = (idobs == iddetection) & (id1 == idinj)

        if not config == None:
            for k,v in config.items():
                setattr(self,k,v)


        # build data cube cascade
        if np.sum(self._mc):
            #data_casc = np.array([Einj[self._mc], Eobs[self._mc],
            #            dt[self._mc], dtheta[self._mc], phi[self._mc], theta[self._mc]])
            data_casc = np.array([Einj[self._mc], Eobs[self._mc],
                        dt[self._mc], phi[self._mc], theta[self._mc]])
        else:
            logging.error("No events pass cascade criterion")
            assert np.sum(self._mc) > 0

        # build data cube for primary spectrum 
        if np.sum(self._mi):
            data_primary = np.array([Einj[self._mi], Eobs[self._mi]])
        else:
            logging.error("No events pass primary spectrum criterion")
            assert np.sum(self._mi) > 0

        # build the histogram
        logging.info("Building the cascade histogram  ...")
        logging.info("Bin shapes: {0}".format(np.array(bins).shape))
        self._hist_casc, self._edges_casc = np.histogramdd(data_casc.T,bins = bins)
        logging.info("Done.")

        logging.info("Building the injected spectrum histogram  ...")
        # TODO: accounted for redshift, but is it correct?
        if np.sum(self._mi):
            if type(config) == dict:
                # bins[0] contain the injected energies
                # bins[0] / ( 1 + z) are the observed energies
                self._hist_prim, self._edges_prim = np.histogramdd(data_primary.T,
                                            bins = (bins[0],bins[0] / (1. + self.Source['z'])))
                                            #bins = (bins[0],bins[0]))
                                            #bins = bins[:2])
            else:
                self._hist_prim, self._edges_prim = np.histogramdd(data_primary.T,bins = (bins[0],bins[1]))
        else:
            logging.error("No events pass primary spectrum criterion and" + \
                " no config file provided, not building primary spectrum")
        logging.info("Done.")


        # get the central bin values
        # for all bins of the cascade spectrum
        self._cen = []
        imax = 3 if tbinslog else 2
        for i,edge in enumerate(self._edges_casc):
            if i < imax: # log bins
                self._cen.append(np.sqrt(edge[:-1] * edge[1:]))
            else: # lin bins
                self._cen.append(0.5 * (edge[:-1] + edge[1:]))

        # get the central bin values
        # for the energies of the injected values
        self._cen_prim = []
        for i,edge in enumerate(self._edges_prim):
            self._cen_prim.append(np.sqrt(edge[:-1] * edge[1:]))

        # build an array for the volume elements
        vol = np.meshgrid(*[np.diff(b) for b in self._edges_casc], indexing = 'ij')
        self._volcube_casc = np.array(vol)

        vol = np.meshgrid(*[np.diff(b) for b in self._edges_prim], indexing = 'ij')
        self._volcube_prim = np.array(vol)

        # 2d array for integration of injected energy
        self._einj = []
        for i,emin in enumerate(self._edges_prim[0][:-1]):
            self._einj.append( np.logspace( np.log10(emin), 
                        np.log10(self._edges_prim[0][i+1]), steps))
        self._einj = np.array(self._einj)

        self._weights_inj = injected[1]
        self._weights = np.ones_like(self._weights_inj)
        return

    @staticmethod
    def gen_from_hd5f(infile, dgrp = 'simEM', roiradius = 3., numebins = 41, degperpix = 0.04, 
                        tbins = None, tbinslog = True):

        """
        Generate Histrogram from hd5f file
        Bin boundaries are set automatically
        to hardcoded reasonable values

        TODO: Let user change binning

        Parameters
        ----------
        infile: str
            path to hd5f file created with the stack_results_lso function

        kwargs
        ------
        roiradius: float
            radius of ROI, in degree, used for binning in phi and theta
            (default: 3.)
        degperpix: float
            with of each pixel in degree for resulting histogram
            which will be used for phi and theta binning
            (default: 0.04, motivated from minimum CTA PSF)
        tbins: None or `~numpy.ndarray`
            bins for time delay, default: None
        tbinslog: bool
            if true, time bins considered to be log spaced
        """
        hfile = h5py.File(infile, 'r+')
        data = hfile[dgrp]
        config = yaml.load(data.attrs['config'])

        bins = []
        # injected energy bins
        bins.append( np.logspace(np.log10(config['Source']['Emin']),
                    np.log10(config['Source']['Emax']),
                    config['Source']['Esteps']))
        # observed energy bins
        bins.append( np.logspace(np.log10(data['E'][()].min()),
                    np.log10(data['E'][()].max()),
                    numebins))
        # time delay
        if tbins is None:

            tmin = np.max([0.1,data['dt'][()].min()])
            bins.append( np.concatenate([[tmin,3.],
                    np.logspace(1.,7,7)]) )
            if bins[-1][-1] < data['dt'][()].max():
                bins[-1] = np.concatenate([bins[-1],[data['dt'][()].max()]])
        else:
            bins.append( tbins )

        # angular separation 
        #bins.append( np.concatenate([[data['dtheta'][()].min()],
                    #np.logspace(-8.,np.log10(80.),21)]))
        #if bins[-1][-1] < data['dtheta'][()].max():
            #bins[-1] = np.concatenate([bins[-1],[data['dtheta'][()].max()]])
        #bins.append( np.linspace(-5.,5.,41) )

        # phi 
        nbins = int(np.ceil(2.*roiradius / degperpix))
        bins.append( np.linspace(-roiradius,roiradius,nbins + 1) )
        # theta 
        #bins.append( np.linspace(90.-5.,90. + 5.,41) )
        bins.append( np.linspace(90.-roiradius,90. + roiradius,nbins + 1) )

        E0 = data['E0'][()]
        E = data['E'][()]
        dt = data['dt'][()]
        #dtheta = data['dtheta'][()]
        phi = data['Protsph'][1,:]
        theta = data['Protsph'][2,:]
        idobs = data['ID'][()]
        id1 = data['ID1'][()]

        injected = np.array([data['intspec/Ecen'][()], data['intspec/weights'][()]])

        hfile.close()
        return EMHist(E0,E,dt,
                   # dtheta,
                    phi,theta,idobs,id1,
                    bins, idinj  = config['Source']['Composition'],
                    iddetection= 22, config = config, injected = injected,
                    tbinslog = tbinslog,
                    degperpix = degperpix)

    @property
    def hist_casc(self):
        return self._hist_casc

    @property
    def hist_prim(self):
        return self._hist_prim

    @property
    def weights(self):
        return self._weights

    @property
    def ecen_primary_obs(self):
        """return the observed central energy of the primary gamma ray spectrum"""
        return self._cen_prim[1]

    @property
    def ecen_primary_inj(self):
        """return the injected central energy of the primary gamma ray spectrum"""
        return self._cen_prim[0]

    @property
    def ecen_casc_obs(self):
        """return the observed central energy of the cascade gamma ray spectrum"""
        return self._cen[1]
    @property
    def ecen_casc_obs_bins(self):
        return self._edges_casc[1]

    @property
    def ecen_casc_inj(self):
        """return the injected central energy of the cascde gamma ray spectrum"""
        return self._cen[0]
    @property
    def ecen_casc_inj_bins(self):
        return self._edges_casc[0]

    @property
    def dt_casc(self):
        """return the central time delay values of the cascde gamma ray spectrum"""
        return self._cen[2]

    @property
    def dt_casc_bins(self):
        return self._edges_casc[2]

    #@property
    #def dtheta_casc(self):
        #"""return the central dtheta values of the cascde gamma ray spectrum"""
        #return self._cen[3]

    @property
    def phi_casc(self):
        """return the central theta values of the cascde gamma ray spectrum"""
        #return self._cen[4]
        return self._cen[3]
    @property
    def phi_casc_bins(self):
        return self._edges_casc[3]

    @property
    def theta_casc(self):
        """return the central theta values of the cascde gamma ray spectrum"""
        #return self._cen[5]
        return self._cen[4]

    @property
    def theta_casc_bins(self):
        return self._edges_casc[4]

    @property
    def injected_spec(self):
        """return the weights of the initially injected spectrum"""
        return self._injected[1]

    def set_weights(self, injspec):
        """
        set weights to compute cascade for an arbitrary spectrum

        Parameters
        ----------
        injspec: function pointer
            function that takes energy in eV and returns flux per energy
            Must be in units per eV
        """
    
        # flux of new injected spectrum integrated in 
        # bins of injected spectrum
        Finj = simps(injspec(self._einj) * self._einj, np.log(self._einj), axis = 1)
        self._weights = Finj / self._weights_inj
        return 

    def apply_weights(self):
        dNcasc_dEtrue_dEobs_dt_dphi_dtheta = (self._hist_casc.T * self._weights).T / \
            np.product(self._volcube_casc, axis = 0)
        dNprim_dEtrue_dEobs = (self._hist_prim.T * self._weights).T / \
            np.product(self._volcube_prim, axis = 0)

        return dNcasc_dEtrue_dEobs_dt_dphi_dtheta, dNprim_dEtrue_dEobs

    def obsspec1D(self, mEinj = [0.,np.nan],
                        mdt = [0.,np.nan],
                       # mdtheta = [0.,np.nan],
                        mphi = [0.,np.nan],
                        mtheta = [0.,np.nan], apply_weights = True):
        """
        Return the observed energy spectrum with cuts 
        along axes applied
        """
        cut = self.apply_weights()[0]
        if np.all(np.isfinite(mtheta)):
            cut = self._hist_casc[...,(self._cen[-1] >= mtheta[0]) & (self._cen[-1] < mtheta[1])]

        if np.all(np.isfinite(mphi)):
            cut = cut[...,(self._cen[-2] >= mphi[0]) & (self._cen[-2] < mphi[1]),:]

        #if np.all(np.isfinite(mdtheta)):
            #cut = cut[...,(self._cen[-3] >= mdtheta[0]) & (self._cen[-3] < mdtheta[1]),:,:]

        if np.all(np.isfinite(mdt)):
            #cut = cut[...,(self._cen[-3] >= mdt[0]) & (self._cen[-3] < mdt[1]),:,:,:]
            cut = cut[...,(self._cen[-3] >= mdt[0]) & (self._cen[-3] < mdt[1]),:,:]
            
        if np.all(np.isfinite(mEinj)):
            cut = cut[(self._cen[0] >= mEinj[0]) & (self._cen[0] < mEinj[1]),...]
        #return cut.sum(axis = (0,2,3,4,5))
        return cut.sum(axis = (0,2,3,4))

    def obsspec3D(self, mEinj = [0.,np.nan],
                        mdt = [0.,np.nan],
                        #mdtheta = [0.,np.nan],
                        apply_weights = True):
        """
        Return the observed spectrum in spatial and energy dimensions with cuts 
        along axes applied
        """
        cut = (self._hist_casc.T * self._weights).T

        if np.all(np.isfinite(mdt)):
            #cut = cut[...,(self._cen[-4] >= mdt[0]) & (self._cen[-4] < mdt[1]),:,:,:]
            cut = cut[...,(self.dt_casc >= mdt[0]) & (self.dt_casc < mdt[1]),:,:]
            
        if np.all(np.isfinite(mEinj)):
            cut = cut[(self._cen[0] >= mEinj[0]) & (self._cen[0] < mEinj[1]),...]

        #if np.all(np.isfinite(mdtheta)):
            #cut = cut[...,(self._cen[-3] >= mdtheta[0]) & (self._cen[-3] < mdtheta[1]),:,:]

        #return cut.sum(axis = (0,2,3))
        return cut.sum(axis = (0,2))

class CRHist(object):
    """
    Class for creation and manipulation 
    of N-dim histogram with CRPropa output for 
    cascades induced by cosmic rays
    """
    def __init__(self, Einj, Eobs, dt,
                   # dtheta,
                    phi, theta, idobs, bins, injected,
                    iddetection= 22, steps = 10, config = None):
        """
        Create the histogram

        Parameters
        ----------
        Einj: `~numpy.ndarray`
            Array with injected energies in eV
        Eobs: `~numpy.ndarray`
            Array with observed energies in eV
        dt: `~numpy.ndarray`
            Array with time delays in years
        phi: `~numpy.ndarray`
            Array with phi coordinates (l or DEC) in degrees
        theta: `~numpy.ndarray`
            Array with theta coordinates (pi / 2. - b or RA) in degrees
        idobs: `~numpy.ndarray`
            Array with observed particle IDs
        bins: list
            list with bin edges for all axis
        injected: `~numpy.ndarray`
            Array with the counts of 
            injected particles


        {options}
       
        iddetection: int or list 
            particle ID for detected particle
            (veto against other particles, default: 22)
        steps: int
            steps for integration of intrinsic spectrum (default: 10)
        """
        # select particles 
        if isinstance(iddetection, int):
            self._mc = (idobs == iddetection)
        elif isinstance(iddetection, Iterable):
            self._mc = idobs == iddetection[0]
            if len(iddetection) > 1:
                for idd in iddetection[1:]:
                    self._mc = self._mc | (idobs == idd) 
        else:
            logging.error("iddetection parameter not understood")

        if not config == None:
            for k,v in config.items():
                setattr(self,k,v)

        # build data cube cascade
        if np.sum(self._mc):
            data_casc = np.array([Einj[self._mc], Eobs[self._mc],
                        dt[self._mc], phi[self._mc], theta[self._mc]])
        else:
            logging.error("No events of selected observed particle type")
            assert np.sum(self._mc) > 0

        # build the histogram
        logging.info("Building the cascade histogram  ...")
        logging.info("Bin shapes: {0}".format(np.array(bins).shape))
        self._hist_casc, self._edges_casc = np.histogramdd(data_casc.T,bins = bins)
        logging.info("Done.")

        # get the central bin values
        # for all bins of the cascade spectrum
        self._cen = []
        for i,edge in enumerate(self._edges_casc):
            if i < 3: # log bins
                self._cen.append(np.sqrt(edge[:-1] * edge[1:]))
            else: # lin bins
                self._cen.append(0.5 * (edge[:-1] + edge[1:]))

        # 2d array for integration of injected energy
        self._einj = []
        for i,emin in enumerate(bins[0][:-1]):
            self._einj.append( np.logspace( np.log10(emin), 
                        np.log10(bins[0][i+1]), steps))
        self._einj = np.array(self._einj)

        self._weights_inj = injected
        self._weights = np.ones_like(self._weights_inj)
        return

    @staticmethod
    def gen_from_hd5f(infile, iddetection, dgrp = 'simEM',
        ebins = 40, roiradius = 3., degperpix = 0.04):
        """
        Generate Histrogram from hd5f file
        Bin boundaries are set automatically
        to hardcoded reasonable values

        TODO: Let user change binning

        Parameters
        ----------
        infile: str or list of strings
            path to hd5f file created with the stack_results_lso function
        iddetection: int
            particle ID for detected particle
            (veto against other particles, default: 22)

        kwargs
        ------
        roiradius: float
            radius of ROI, in degree, used for binning in phi and theta
            (default: 3.)
        degperpix: float
            with of each pixel in degree for resulting histogram
            which will be used for phi and theta binning
            (default: 0.04, motivated from minimum CTA PSF)
        ebins: int
            number of energy bins for cascade energy axis
            (default: 40)
        """
        if isinstance(infile, str):
            files = [infile]
        elif isinstance(infile, Iterable): 
            files = infile

        for i,f in enumerate(files):
            hfile = h5py.File(f, 'r+')
            data = hfile[dgrp]
            config = yaml.load(data.attrs['config'])

            if not i:
                Ebins_inj = np.logspace(np.log10(config['Source']['Emin']),
                    np.log10(config['Source']['Emax']),
                    config['Source']['Esteps'])
                E0 = data['E0'][()]
                E = data['E'][()]
                dt = data['dt'][()]
                #dtheta = data['dtheta'][()]
                phi = data['Protsph'][1,:]
                theta = data['Protsph'][2,:]
                idobs = data['ID'][()]
                weights = data['intspec/weights'][()]
            else:
                ebi = np.logspace(np.log10(config['Source']['Emin']),
                    np.log10(config['Source']['Emax']),
                    config['Source']['Esteps'])
                if not ebi[0] == Ebins_inj[-1]:
                    raise Exception("Bins not sorted: left bin edge energy bin" \
                    "of new file must be equal to right bin edge of previous file")
                Ebins_inj = np.concatenate([Ebins_inj, ebi[1:]])
                E = np.concatenate([E,data['E'][()]])
                E0 = np.concatenate([E0,data['E0'][()]])
                dt = np.concatenate([dt,data['dt'][()]])
                phi = np.concatenate([phi,data['Protsph'][1,:]])
                theta = np.concatenate([theta,data['Protsph'][2,:]])
                idobs = np.concatenate([idobs,data['ID'][()]])
                weights = np.concatenate([weights,data['intspec/weights'][()]])

            hfile.close()

        bins = []
        # injected energy bins
        bins.append( Ebins_inj )
        # observed energy bins
        bins.append( np.logspace(np.log10(E.min()),
                    np.log10(E.max()),
                    ebins ))
        # time delay
        mint = np.floor(np.log10(dt.min()))
        maxt = np.ceil(np.log10(dt.max()))
        if np.isnan(maxt):
            maxt = 8.
        if np.isnan(mint):
            mint = 0.
        bins.append( np.logspace(mint,maxt,int(maxt - mint) + 1) )

        nbins = int(np.ceil(2.*roiradius / degperpix))
        # phi 
        bins.append( np.linspace(-roiradius,roiradius,nbins) )
        # theta 
        bins.append( np.linspace(90.-roiradius,90. + roiradius,nbins) )

        return CRHist(E0,E,dt,
                    phi,theta,idobs,
                    bins,
                    iddetection= iddetection,
                    config = config, injected = weights)

    @property
    def hist_casc(self):
        return self._hist_casc

    @property
    def weights(self):
        return self._weights

    @property
    def ecen_inj(self):
        """return the injected central energy of the cosmic ray spectrum"""
        return self._cen[0]

    @property
    def ecen_casc_obs(self):
        """return the observed central energy of the cascade gamma ray spectrum"""
        return self._cen[1]
    @property
    def ecen_casc_obs_bins(self):
        return self._edges_casc[1]

    @property
    def ecen_casc_inj(self):
        """return the injected central energy of the cascde gamma ray spectrum"""
        return self._cen[0]
    @property
    def ecen_casc_inj_bins(self):
        return self._edges_casc[0]

    @property
    def dt_casc(self):
        """return the central time delay values of the cascde gamma ray spectrum"""
        return self._cen[2]

    @property
    def dt_casc_bins(self):
        return self._edges_casc[2]

    #@property
    #def dtheta_casc(self):
        #"""return the central dtheta values of the cascde gamma ray spectrum"""
        #return self._cen[3]

    @property
    def phi_casc(self):
        """return the central theta values of the cascde gamma ray spectrum"""
        #return self._cen[4]
        return self._cen[3]
    @property
    def phi_casc_bins(self):
        return self._edges_casc[3]

    @property
    def theta_casc(self):
        """return the central theta values of the cascde gamma ray spectrum"""
        #return self._cen[5]
        return self._cen[4]

    @property
    def theta_casc_bins(self):
        return self._edges_casc[4]

    @property
    def injected_spec(self):
        """return the weights of the initially injected spectrum"""
        return self._weights_inj

    def set_weights(self, injspec):
        """
        set weights to compute cascade for an arbitrary spectrum

        Parameters
        ----------
        injspec: function pointer
            function that takes energy in eV and returns flux per energy
            Must be in units per eV
        """
    
        # flux of new injected spectrum integrated in 
        # bins of injected spectrum
        Finj = simps(injspec(self._einj) * self._einj, np.log(self._einj), axis = 1)
        self._weights = Finj / self._weights_inj
        return 

    def obsspec1D(self, mEinj = [0.,np.nan],
                        mdt = [0.,np.nan],
                        mphi = [0.,np.nan],
                        mtheta = [0.,np.nan], apply_weights = True):
        """
        Return the observed energy spectrum with cuts 
        along axes applied
        """
        if apply_weights:
            cut = (self._hist_casc.T * self._weights).T
        else:
            cut = self._hist_casc

        if np.all(np.isfinite(mtheta)):
            cut = self._hist_casc[...,(self._cen[-1] >= mtheta[0]) & (self._cen[-1] < mtheta[1])]

        if np.all(np.isfinite(mphi)):
            cut = cut[...,(self._cen[-2] >= mphi[0]) & (self._cen[-2] < mphi[1]),:]

        if np.all(np.isfinite(mdt)):
            cut = cut[...,(self._cen[-3] >= mdt[0]) & (self._cen[-3] < mdt[1]),:,:]
            
        if np.all(np.isfinite(mEinj)):
            cut = cut[(self._cen[0] >= mEinj[0]) & (self._cen[0] < mEinj[1]),...]
        return cut.sum(axis = (0,2,3,4))

    def obsspec3D(self, mEinj = [0.,np.nan],
                        mdt = [0.,np.nan],
                        apply_weights = True):
        """
        Return the observed spectrum in spatial and energy dimensions with cuts 
        along axes applied
        """
        cut = (self._hist_casc.T * self._weights).T

        if np.all(np.isfinite(mdt)):
            cut = cut[...,(self.dt_casc >= mdt[0]) & (self.dt_casc < mdt[1]),:,:]
            
        if np.all(np.isfinite(mEinj)):
            cut = cut[(self._cen[0] >= mEinj[0]) & (self._cen[0] < mEinj[1]),...]

        return cut.sum(axis = (0,2))

class emmap(object):
    """
    class for creation and manipulation
    of n-dim gammapy wcs map with crpropa output for electromagnetic
    cascade and histogram for intrinsic spectrum

    use this if the temporal information is not important
    and you only care about the emission that has arrived
    at earth with a delay < tmax.
    """
    def __init__(self, values, edges, skycoord,
                    injected = None,
                    idinj  = 22,
                    iddetection= 22, config = None,
                    tmax = 1e6,
                    steps = 10,
                    binsz = 0.04, width = 6.):
        """
        create the histogram

        parameters
        ----------
        values: dict
            dictionary with values of crpropa simulation
        edges: dict
            dictionary with bin edges
        skycoord: `~astropy.coordinates.skycoord` object
            sky coordinates for observation

        {options}

        tmax : float
            maximally allowed time delay in years (default: 1e6)
        injected: `~numpy.ndarray`
            (2xn) dim array with the central energies and counts of
            injected particles
        idinj: int
            particle id of injected particles
            (used to determine if particle is produced in cascade, default: 22)
        iddetection: int
            particle id for detected particle
            (veto against other particles, default: 22)
        steps: int
            steps for integration of intrinsic spectrum (default: 10)
        binsz: float
            assumed pixelization of skymaps in degrees / pix
        width: float
            assumed width of skymaps in degrees
        config: dict or none
            simulation dict
        """

        self._centers = ordereddict()
        self._widths = ordereddict()
        self._edges = edges
        self._skycoord = skycoord
        self._binsz = binsz
        self._roiwidth = width

        # get the central bin values
        # for all bins of the cascade spectrum
        # as well as the central widths
        for k,v in edges.items():
            self._widths[k] = np.diff(edges[k])
            if k == 'lon' or k == 'lat': # lin bins
                self._centers[k] = 0.5 * edges[k][1:] * edges[k][:-1]
            else: # log bins
                self._centers[k] = np.sqrt(edges[k][1:] * edges[k][:-1])

        self._injected = injected
        # cascade condition
        self._mc = (values['idobs'] == iddetection) & \
                    (values['id1'] != idinj) & \
                    (values['t_delay'] <= tmax)
        # injected spectrum condition
        self._mi = (values['idobs']== iddetection) & (values['id1'] == idinj)

        if not config == None:
            for k,v in config.items():
                setattr(self,k,v)

        # build data cube cascade
        if np.sum(self._mc):
            keys = ['etrue', 'eobs', 'lon', 'lat']
            data_casc = np.array([values[k][self._mc] \
                                    for k in keys])
            edges_casc = [edges[k] for k in keys]
        else:
            logging.error("no events pass cascade criterion")
            assert np.sum(self._mc) > 0

        # build data cube for primary spectrum
        if np.sum(self._mi):
            keys = ['etrue', 'eobs']
            data_primary = np.array([values[k][self._mi] \
                                    for k in keys])
            edges_primary = [edges[k] for k in keys]

        else:
            raise exception("no events pass primary spectrum criterion")

        # build the histogram
        logging.info("building the cascade histogram" \
                        "for tmax = {0:.2e} years ...".format(tmax))
        logging.info("bin shapes: {0}".format([edges_casc[i].shape for i in range(len(edges_casc))]))
        self._hist_casc, self._edges_casc = np.histogramdd(data_casc.t,bins = edges_casc)
        logging.info("done.")

        logging.info("building the injected spectrum histogram  ...")
        # todo: accounted for redshift, but is it correct?
        if np.sum(self._mi):
            if isinstance(config, dict):
                # bins[0] contain the injected energies
                # bins[0] / ( 1 + z) are the observed energies
                self._hist_prim, self._edges_prim = np.histogramdd(data_primary.t,
                                            bins = (edges_primary[0],
                                                    edges_primary[0] / (1. + self.source['z'])))
            else:
                self._hist_prim, self._edges_prim = np.histogramdd(data_primary.t,
                                                        bins = (edges_primary[0],edges_primary[1]))
            self._centers['eobs_prim'] = np.sqrt(self._edges_prim[1][1:] * \
                                    self._edges_prim[1][:-1] )
            self._widths['eobs_prim'] = np.diff(self._edges_prim[1])

        else:
            raise exception("no events pass primary spectrum criterion")
        logging.info("done.")

        # build the maps
        eobs_casc_axis = mapaxis.from_edges(edges['eobs'],
                            interp='log', name = 'eobs')
        self._casc_map = map.create(binsz = binsz, width = width,
                                skydir = skycoord,
                                axes = [eobs_casc_axis])

        # create a map in which each pixel
        # contains the volume of the pixel
        self._vol_map = map.create(binsz = binsz, width = width,
                                skydir = skycoord,
                                axes = [eobs_casc_axis])
        self._solid_angle = map.create(binsz = binsz, width = width,
                                skydir = skycoord,
                                axes = [eobs_casc_axis])

        # create a map with angular separations
        self._sep_map = map.create(
                binsz=binsz,
                map_type='wcs',
                width=width,
                skydir = skycoord)

        # fill the map with the separation from center
        lon, lat = self._sep_map.geom.get_coord()
        c = skycoord(lon, lat, unit = 'deg', frame = skycoord.frame)
        sep = skycoord.separation(c).value
        self._sep_map.set_by_coord((lon,lat), sep)

        we, wlon, wlat = np.meshgrid(self._widths['eobs'],
                            np.radians(self._widths['lon']),
                            np.radians(self._widths['lat']),
                            indexing = 'ij')

        # volume map in units of sr * ev
        self._vol_map.set_by_idx(self._vol_map.geom.get_idx(),
                                    we * wlon * wlat)

        # solid angle map
        self._solid_angle.set_by_idx(self._solid_angle.geom.get_idx(),
                                    wlon * wlat)

        # fill the cascade map
        # make it in units photons / s / cm^2 / ev / sr
        # s and cm^2 come from weights later
        il, ib, ie = self._casc_map.geom.get_idx()
        self._casc_map.set_by_idx(self._casc_map.geom.get_idx(),
                                self._hist_casc.sum(axis = 0)[ie,il,ib] \
                                / we / wlon / wlat)



        # map with primary gamma ray emission
        eobs_prim_axis = mapaxis.from_edges(self._edges_prim[1],
                            interp = 'log', name = 'eobs')

        self._prim_map = map.create(binsz = binsz, width = width,
                                skydir = skycoord,
                                axes = [eobs_prim_axis])

        self._idx_prim_cen = [[self._prim_map.geom.center_pix[0]],
                               [self._prim_map.geom.center_pix[1]],
                               range(self._prim_map.data.shape[0])]

        # make it in units photons / s / cm^2 / ev / sr
        self.__fprim = self._widths['eobs_prim'] * \
                    np.radians(self._widths['lon'][0]) * \
                    np.radians(self._widths['lat'][0])
        self._prim_map.set_by_idx(self._idx_prim_cen,
                self._hist_prim.sum(axis = 0)[np.newaxis, np.newaxis, :] \
                / self.__fprim)

        # 2d array for integration of injected energy
        self._einj = []
        for i,emin in enumerate(self._edges['etrue'][:-1]):
            self._einj.append( np.logspace( np.log10(emin),
                        np.log10(self._edges['etrue'][i+1]), steps))
        self._einj = np.array(self._einj)

        self._weights = np.ones_like(self._centers['etrue'])
        self._weights_inj = injected[1]

        return

    @staticmethod
    def gen_from_hd5f(infile, skycoord,
            dgrp = 'simem',
            width = 6.,
            ebins = 41,
            binsz = 0.04, tmax = 1e6):
        """
        generate gammapy.map from hd5f file
        bin boundaries are set automatically
        to hardcoded reasonable values

        parameters
        ----------
        infile: str
            path to hd5f file created with the stack_results_lso function

        skycoord: `~astropy.coordinates.skycoord` object
            sky coordinates for observation


        kwargs
        ------
        width: float
            width of roi, in degrees, used for binning in phi and theta
            (default: 3.)
        binsz: float
            with of each pixel in degrees for resulting histogram
            which will be used for phi and theta binning
            (default: 0.04, motivated from minimum cta psf)
        ebins: int
            total number of bins of observed energy
        """
        hfile = h5py.file(infile, 'r+')
        data = hfile[dgrp]
        config = yaml.load(data.attrs['config'])

        edges = ordereddict({})
        # injected energy bins
        edges['etrue'] = np.logspace(np.log10(config['source']['emin']),
                    np.log10(config['source']['emax']),
                    config['source']['esteps'])
        # observed energy bins
        edges['eobs'] = np.logspace(np.log10(data['e'][()].min()),
                    np.log10(data['e'][()].max()),
                    ebins)
        # time delay
        tmin = np.max([0.1,data['dt'][()].min()])

        edges['t_delay'] = np.concatenate([[tmin,3.],
                    np.logspace(1.,7,7)])

        if edges['t_delay'][-1] < data['dt'][()].max():
            edges['t_delay'] = np.concatenate([edges['t_delay'],[data['dt'][()].max()]])

        # phi
        nbins = int(np.ceil(2.*width/2./binsz))
        edges['lon'] =  np.linspace(-width/2.,width/2.,nbins + 1)
        # theta
        edges['lat'] =  np.linspace(90.-width/2.,90. + width/2.,nbins + 1)

        values = {}
        values['etrue'] = data['e0'][()]
        values['eobs'] = data['e'][()]
        values['t_delay'] = data['dt'][()]
        values['lon'] = data['protsph'][1,:]
        values['lat'] = data['protsph'][2,:]
        values['idobs'] = data['id'][()]
        values['id1'] = data['id1'][()]

        injected = np.array([data['intspec/ecen'][()], data['intspec/weights'][()]])

        hfile.close()
        return emmap(values, edges, skycoord,
                    injected = injected,
                    idinj  = config['source']['composition'],
                    iddetection= 22, config = config,
                    binsz = binsz, width = width, tmax = tmax)

    def make_central_coordinates(self, energies, name = 'eobs'):
        """
        make a coordinate grid for the central
        pixel of the spatial dimensions
        with an additional energy axis

        parameter
        ---------
        energies: `~numpy.ndarray`
            array with energies for the energy coordinate axis

        returns
        -------
        `~gammapy.mapcoord` object with the coordinates
        of central spatial pixel and energies
        """
        # get the coordinates of the cental spatial pixel
        # and all indeces in energy
        central_lon = self._casc_map.geom.center_coord[0]
        central_lat = self._casc_map.geom.center_coord[1]

        return mapcoord.create({'lon' : central_lon,
                                'lat' : central_lat,
                                name : energies})

    def make_full_map_coordinates(self, energies, name = 'eobs'):
        """
        make a coordinate grid for all
        pixels of the spatial dimensions
        with an additional energy axis

        parameter
        ---------
        energies: `~numpy.ndarray`
            array with central energies for the energy coordinate axis

        returns
        -------
        `~gammapy.mapcoord` object with the coordinates
        of central spatial pixel and energies
        """
        ll,bb =  self._casc_map.geom.get_coord()[:-1]
        ll = ll[0]
        bb = bb[0]
        lll = np.vstack([ll[np.newaxis, ...] for i in range(energies.size) ])
        bbb = np.vstack([bb[np.newaxis, ...] for i in range(energies.size) ])
        eee = np.vstack([e * np.ones_like(ll)[np.newaxis,...] for e in energies])

        return mapcoord.create({'lon' : lll,
                                'lat' : bbb,
                                name : eee})

    def set_weights(self, injspec):
        """
        set weights to compute cascade for an arbitrary spectrum

        parameters
        ----------
        injspec: function pointer
            function that takes energy in ev and returns flux per energy
            must be in units per ev
        """

        # flux of new injected spectrum integrated in
        # bins of injected spectrum
        finj = simps(injspec(self._einj) * self._einj, np.log(self._einj), axis = 1)
        # update weights
        self._weights = finj / self._weights_inj
        return

    def apply_weights(self, injspec):
        """
        apply the weights to the maps

        parameters
        ----------
        injspec: function pointer
            function that takes energy in ev and returns flux per energy
            must be in units per ev
        """
        self.set_weights(injspec)
        cut = (self._hist_casc.t * self._weights).t
        il, ib, ie = self._casc_map.geom.get_idx()
        self._casc_map.set_by_idx(self._casc_map.geom.get_idx(),
                                cut.sum(axis = 0)[ie, il, ib] / \
                                self._vol_map.get_by_idx((il,ib,ie)))

        self._prim_map.set_by_idx(self._idx_prim_cen,
                (self._hist_prim.t * self._weights).t.sum(axis = 0)[np.newaxis, np.newaxis, :] \
                / self.__fprim)
        return
