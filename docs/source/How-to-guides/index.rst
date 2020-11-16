.. role:: bash(code)
   :language: bash

Guides
===========================================

In this section, details on the required inputs and optional inputs of the METOD algorithm are provided.

All source code for the METOD algorithm can be found `here <https://github.com/Megscammell/METOD-Algorithm/tree/master/src/metod>`_. To use the METOD algorithm, the metod.py program needs to be executed. Inputs of metod.py are as follows. ::

   def metod(f, g, func_args, d, num_points=1000, beta=0.01,
             tolerance=0.00001, projection=False, const=0.1, m=3,
             option='minimize_scalar', met='Brent', initial_guess=0.05,
             set_x=np.random.uniform, bounds_set_x=(0, 1),
             no_inequals_to_compare='All', usage='metod_algorithm',
             relax_sd_it=1)

Input parameters :bash:`f`, :bash:`g`, :bash:`func_args` and :bash:`d` are required inputs and the remaining input parameters are optional inputs.

.. toctree::
   :maxdepth: 2

   Required_inputs
   Optional_inputs
   Usage
