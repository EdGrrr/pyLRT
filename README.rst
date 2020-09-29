pyLRT - A simple python interface for LibRadTran
************************************************

Edward Gryspeerdt - Space and Atmospheric Physics Group Imperial College London, 2020

A simple python interface/wrapper for LibRadTran.

Note that you will need to download LibRadTran separately from http://www.libradtran.org/doku.php

Features
========

* A simple class for managing a set of options for UVSPEC
* Can parse UVSPEC output into an xarray
* Parses verbose output (so easy to use pre-calculated optical properties)
* Includes a set of examples (for a set of atmospheric radiation lectures)
   
Setup
=====

Run ``python setup.py install``

To make use of the function ``get_lrt_folder()``, place the path to your libradtran folder in the file ``~/.pylrtrc``


Usage
=====

::

   from pyLRT import RadTran, get_lrt_folder

   LIBRADTRAN_FOLDER = get_lrt_folder()

   slrt = RadTran(LIBRADTRAN_FOLDER)
   slrt.options['rte_solver'] = 'disort'
   slrt.options['source'] = 'solar'
   slrt.options['wavelength'] = '200 2600'

   output = slrt.run(verbose=True)
 
For more advanced examples, please see the examples directory.
