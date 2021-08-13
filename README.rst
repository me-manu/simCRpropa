simCRpropa
==========

Python wrapper for CRPropa3 to simulate electromagnetic cascades in the intergalactic medium
produced through gamma rays or cosmic rays interacting with background radiation fields.
An interace with the analysis of Fermi-LAT and IACT data is provided through gammapy.

Prerequisites
-------------

Python 2.7 or and the following packages:
    - A running version of CRPropa3, version 3.1.6 or higher
    - hdf5py
    - numpy 
    - scipy
    - astropy


For running the combined Fermi-LAT and IACT analysis, you will need python 3 and 
gammapy version 0.18.2 or later.


Installation
------------

Currtently, no installation through package managers is supported. Please clone / fork the repository 
and add the file path to your `PYTHONPATH` variable.

Getting Started
---------------

Please take a look at the example notebooks provided in the `notebooks/` folder.

Acknowledgements
----------------

This development of this package has received support from the European Research Council (ERC) under
the European Union's Horizon 2020 research and innovation program Grant agreement No. 843800 (GammaRayCascades).

License
-------

This project is licensed under a 3-clause BSD style license - see the
``LICENSE`` file.
