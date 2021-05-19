.. role:: bash(code)
   :language: bash

Outputs of the METOD algorithm
========================================

List of available results
--------------------------

As discussed in :ref:`Example 1<ex1>` and :ref:`Example 2<ex2>`, outputs can be saved to csv files.

The table below contains details of the outputs from :bash:`metod.py`.

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
     - Number of duplicate minimizers :math:`x_j^{(K_j)}` (see :eq:`sd3` for more details).
   * - :bash:`starting_points`
     - list
     - Starting points used by the METOD algorithm.
