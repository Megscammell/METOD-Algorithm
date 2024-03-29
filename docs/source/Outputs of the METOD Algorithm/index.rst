.. role:: bash(code)
   :language: bash

Outputs of the METOD Algorithm
========================================

List of available results
--------------------------

As discussed in :ref:`Example 1<ex1>` and :ref:`Example 2<ex2>`, outputs can be saved to csv files.

The table below contains details of the outputs from the METOD Algorithm.

.. list-table:: Outputs of metod.py
   :widths: 25 25 50
   :header-rows: 1

   * - Ouput
     - Type
     - Description
   * - :bash:`unique_minimizers`
     - list
     - Each unique minimizer found.
   * - :bash:`unique_number_of_minimizers`
     - integer
     - Total number of unique minimizers found.
   * - :bash:`func_vals_of_minimizers`
     - list
     - Function evaluated at each unique minimizer.
   * - :bash:`excessive_descents`
     - integer
     - Total number of repeated local descents to the same local minimizer.
   * - :bash:`starting_points`
     - list
     - Starting points used by the METOD Algorithm.
   * - :bash:`no_grad_evals`
     - 1-D array
     - Total number of gradient evaluations computed by the METOD algorithm for each starting point.
       If local descent is terminated early for a point, the total number of gradient evaluations will
       be small.
