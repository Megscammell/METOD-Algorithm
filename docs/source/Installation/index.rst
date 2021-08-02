.. role:: bash(code)
   :language: bash

.. _installation:

Installation
=============
To install the METOD Algorithm from source:

.. code-block:: bash

   $ git clone https://github.com/Megscammell/METOD-Algorithm.git
   $ cd METOD-Algorithm
   $ python setup.py install

To ensure all tests are working, create an environment and run the tests using :bash:`pytest`:

.. code-block:: bash

   $ conda env create -f environment.yml
   $ conda activate metod_algorithm
   $ pytest


An example of applying the METOD Algorithm with an objective function and gradient is presented below:

.. code-block:: python
  :linenos:

   >>> import numpy as np
   >>> import math
   >>> import metod_alg as mt
   >>>
   >>> np.random.seed(90)
   >>> d = 2
   >>> A = np.array([[1, 0],[0, 10]])
   >>> theta = np.random.uniform(0, 2 * math.pi)
   >>> rotation = np.array([[math.cos(theta), -math.sin(theta)],
   ...                     [math.sin(theta), math.cos(theta)]])
   >>> x0 = np.array([0.5, 0.2])
   >>>
   >>> def f(x, x0, A, rotation):
   ...     return 0.5 * (x - x0).T @ rotation.T @ A @ rotation @ (x - x0)
   ...
   >>> def g(x, x0, A, rotation):
   ...     return rotation.T @ A @ rotation @ (x - x0)
   ...
   >>> args = (x0, A, rotation)
   >>> (discovered_minimizers,
   ...  number_minimizers,
   ...  func_vals_of_minimizers,
   ...  excessive_no_descents, 
   ...  starting_points,
   ...  no_grad_evals) = mt.metod(f, g, args, d, num_points=10)
   >>> assert(np.all(np.round(discovered_minimizers[0], 3) == np.array([0.500, 0.200])))
   >>> assert(number_minimizers == 1)
   >>> assert(np.round(func_vals_of_minimizers, 3) == 0)
   >>> assert(excessive_no_descents == 0)
   >>> assert(np.array(starting_points).shape == (10, d))

The purpose of each line of code within the example is discussed in the following table.

.. list-table::
   :widths: 10 50
   :header-rows: 1

   * - Line number
     - Purpose of each line of code within the example
   * - 1 - 3
     - Import the required libraries. 
   * - 5
     - Initialize the pseudo-random number generator seed.
   * - 6
     - Set the dimension as :bash:`d = 2`.	
   * - 7
     -  Create the variable :bash:`A`, which is assigned a diagonal matrix. 
   * - 8
     - Create the variable :bash:`theta`, which is assigned a value chosen uniformly at random from :math:`[0, 2\pi]`.
   * - 9 - 10
     - Create the variable :bash:`rotation`, which is assigned a rotation matrix using :bash:`theta`.
   * - 11
     - Create the variable :bash:`x0`, which is the minimizer of :bash:`f`. 
   * - 13 - 14
     -  Define a function :bash:`f` to apply the METOD Algorithm.
   * - 16-17
     - Define the gradient :bash:`g`, which returns the gradient of :bash:`f`.
   * - 19
     - Set :bash:`x0`, :bash:`A` and :bash:`rotation` as objective function arguments. The function arguments are required to run :bash:`f` and :bash:`g`. 
   * - 20 - 25
     - Run the METOD Algorithm with :bash:`f`, :bash:`g`, :bash:`args`, :bash:`d` and optional input :bash:`num_points=10` to obtain the METOD Algorithm outputs.
   * - 26 - 30
     - Check outputs of the METOD Algorithm.
