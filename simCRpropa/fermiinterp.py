import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
from scipy.interpolate import UnivariateSpline as USpline
from collections import OrderedDict


class LogLikeCubeFermi(object):
    """
    Class to perform auxiliary operations on
    Fermi-LAT likelihood cube
    """

    def __init__(self, filename):
        """
        Initialize the class

        Parameters
        ----------
        filename: str
        path to npy file generated from Fermi analysis
        which contains a dict with keys for each source and values are the likelihood tables
        """
        self._data = np.load(filename, allow_pickle=True, encoding='latin1').flat[0]
        self._llh = None
        self._grids = None
        self._params = None
        self._norms = None
        self._interp = None
        self._log_norm_array = None
        self._llh_grid = None

    @property
    def params(self):
        return self._params

    @property
    def log_norm_array(self):
        return self._log_norm_array

    @property
    def interp(self):
        return self._interp

    @staticmethod
    def reshape_cube(table, table_key='loglike', keys=('B', 'maxTurbScale', 'th_jet', 'Cutoff', 'Index')):
        """Reshape a flat log l array to a cube """
        params, idx = OrderedDict(), OrderedDict()

        for k in keys:
            params[k], idx[k] = np.unique(table[k].data, return_inverse=True)

        g = np.meshgrid(*[v for v in params.values()], indexing='ij')
        grids = OrderedDict()
        for i, k in enumerate(keys):
            grids[k] = g[i]

        # new shape is shape of parameters + source norm + halo norm
        new_shape = list(grids[keys[0]].shape) + list(table[table_key].data.shape[1:])
        result = table[table_key].data.reshape(new_shape)

        # check that new shape is correct excluding halo and source norm
        for k in keys:
            assert np.sum(table[k].reshape(result.shape[:len(grids)]) - grids[k]) == 0.

        return new_shape, result, grids, params

    def get_llh_one_source(self, src, norm_key='dnde_src'):
        """
        Get the reshaped likelihood for one source

        Parameters
        ----------
        src: str
            source name
        """
        _, self._llh, self._grids, self._params = self.reshape_cube(self._data[src], table_key='loglike')
        _, self._norms, _, _ = self.reshape_cube(self._data[src], table_key=norm_key)

    def interp_llh(self, B, l_coh, th_jet, idx_halo=-1, **interp_kwargs):
        """
        Interpolate log likelihood cube for a particular set of
        simulation parameters

        Parameters
        ----------
        B: float
            IGMF strength for which the logl cube is interpolated

        l_coh:
            coherence length for which the logl cube is interpolated

        th_jet:
            theta jet value for which the logl cube is interpolated

        idx_halo: int or "profile"
            halo normalization to be used.
        """
        interp_kwargs.setdefault('method', 'nearest')
        interp_kwargs.setdefault('fill_value', None)
        interp_kwargs.setdefault('bounds_error', False)

        values = {"B": B, "th_jet": th_jet, "maxTurbScale": l_coh}

        if self._llh is None:
            raise ValueError("No logl cube initialized, run get_llh_one_source function first")

        # select a sub cube with the right parameters
        idx = {}
        for k in ["B", "maxTurbScale", "th_jet"]:
            idx[k] = np.where(self._params[k] == values[k])[0]
            if not len(idx):
                raise ValueError("{0:s} not in list: {1}".format(values[k], self._params[k]))

        llh = self._llh[idx["B"][0], idx["maxTurbScale"][0], idx["th_jet"][0]]
        norms = self._norms[idx["B"][0], idx["maxTurbScale"][0], idx["th_jet"][0]]

        # select how you treat the halo normalization
        if isinstance(idx_halo, str) and idx_halo == "profile":
            # profile over halo normalization
            llh = llh.max(axis=-1)
        elif isinstance(idx_halo, int):
            llh = llh[..., idx_halo]
        else:
            raise ValueError("idx_halo not understood")

        # now llh has shape Cutoff x Index x source normalization
        # but for each Cutoff and Index, normalization array is different
        # therefore, perform first a piece-wise interpolation for each
        # combination of cutoff and index over the same norm array
        self._log_norm_array = np.linspace(np.log10(norms.min()),
                                           np.log10(norms.max()),
                                           int(norms.shape[-1] * 4. / 3.))

        self._index_array= np.linspace(-0.5, 5., 50)

        self._llh_grid = np.zeros((self._params["Cutoff"].size,
                                   self._params["Index"].size,
                                   self._log_norm_array.size))

        # first pass: bring likelihood cube to regular grid over norms
        for i, cut in enumerate(self._params['Cutoff']):
            for j, ind in enumerate(self._params['Index']):

                # for some reason, sometimes same normalization values are stored, 
                # this let's the interpolation crash
                if not np.all(np.diff(np.log10(norms[i,j])) > 0.):
                    mcut = np.diff(np.log10(norms[i,j])) > 0.
                    norms_increasing = np.insert(norms[i,j][1:][mcut], 0, norms[i,j][0])
                    llh_increasing = np.insert(llh[i,j][1:][mcut], 0, llh[i,j][0])
                    idxs = np.argsort(norms_increasing)
                    spline = USpline(np.log10(norms_increasing[idxs]),
                                     llh_increasing[idxs],
                                     k=2, s=0, ext='extrapolate')


                else:
                    spline = USpline(np.log10(norms[i, j]), llh[i, j], k=2, s=0, ext='extrapolate')
                self._llh_grid[i, j] = spline(self._log_norm_array)

        self._llh_grid_extend = np.zeros((self._params["Cutoff"].size,
                                          self._index_array.size,
                                          self._log_norm_array.size))

        # second pass: extend likelihood over index with 2D spline interpolation
        for i, cut in enumerate(self._params['Cutoff']):
            for j, norm in enumerate(self._log_norm_array):
                spline = USpline(self._params["Index"], self._llh_grid[i, :, j], k=2, s=0, ext='extrapolate')
                self._llh_grid_extend[i, :, j] = spline(self._index_array)

        # now perform the grid interpolation
        #self._interp = RegularGridInterpolator(points=(self._params["Cutoff"],
        #                                               self._params["Index"],
        #                                               self._log_norm_array),
        #                                       values=self._llh_grid,
        #                                       **interp_kwargs
        #                                       )
        self._interp = RegularGridInterpolator(points=(self._params["Cutoff"],
                                                       self._index_array,
                                                       self._log_norm_array),
                                               values=self._llh_grid_extend,
                                               **interp_kwargs
                                               )

