Installation
===============================

Dependencies:
-------------

CSI relies on a lot of amazing librairies written by smart people. Please install:

- python3
- gcc
- numpy
- spicy
- shapely
- pyproj
- matplotlib
- cartopy
- multiprocessing
- h5py
- okada4py (available on `GitHub <https://github.com/jolivetr/okada4py>`_)

CSI also has ties with EDKS, a software written by Luis Rivera (Univ. Strasbourg). Since this is not mine, I cannot distribute it. It allows to compute Green's functions in a layered medium. Other softwares can be used but we haven't implmemented the links with CSI.

Repositories:
-------------

CSI is available on `GitHub <https://github.com/jolivetr/csi>`_! Go get it and clone it to your computer.

Install:
--------

There is nothing to compile for CSI. It is pure python and we haven't written a proper install script.
Therefore, the easiest way to go is to add the directory where you have cloned CSI to your PYTHONPATH environment variable:

For instance, in Bash, add to your .bashrc or .bash_profile:

>> export PYTHONPATH=/where/I/did/drop/the/code:$PYTHONPATH

This should do it!

