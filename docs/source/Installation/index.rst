.. role:: bash(code)
   :language: bash

.. _installation:

Installation
=============
To install the METOD Algorithm from source: ::

   $ git clone https://github.com/Megscammell/METOD-Algorithm.git
   $ cd METOD-Algorithm
   $ python setup.py install

To ensure all tests are working, create an environment and run the tests using :bash:`pytest`: ::

   $ conda env create -f environment.yml
   $ conda activate metod_algorithm
   $ pytest


.. role:: bash(code)
   :language: python

An example of applying the METOD algorithm with an objective function and gradient is presented below :
::

   >>> import numpy as np
   >>> import math
   >>> import metod as mt
   >>> from metod import objective_functions as mt_obj
   >>> np.random.seed(90)
   >>> f = mt_obj.single_quad_function
   >>> g = mt_obj.single_quad_gradient 
   >>> d = 2
   >>> theta = np.random.uniform(0, 2 * math.pi)
   >>> rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
   >>> A = np.array([[1, 0], [0, 10]])
   >>> x0 = np.array([0.5, 0.2])
   >>> args = (x0, A, rotation)
   >>> (discovered_minimizers, number_minimizers,
   ...  func_vals_of_minimizers,
   ...  excessive_no_descents, 
   ...  starting_points) = mt.metod(f, g, args, d, num_points=10)
   >>> assert(np.all(np.round(discovered_minimizers[0], 3) == np.array([0.500,0.200])))
   >>> assert(number_minimizers == 1)
   >>> assert(np.round(func_vals_of_minimizers , 3) == 0)
   >>> assert(excessive_no_descents == 0)
   >>> assert(np.array(starting_points).shape == (10, d))