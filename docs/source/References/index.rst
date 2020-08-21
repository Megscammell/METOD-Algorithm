References
==========

List of available results
--------------------------

As discussed in :ref:`Example 1<ex1>` and :ref:`Example 2<ex2>`, outputs can be saved to csv files.

The table below contains details of the outputs from metod.py.

.. list-table:: Outputs of metod.py
   :widths: 25 25 50
   :header-rows: 1

   * - Ouput
     - Type
     - Description
   * - unique_minima
     - list
     - Each unique minimizer found.
   * - unique_number_of_minima
     - integer
     - Total number of unique minimizers found.
   * - func vals of minimas
     - list
     - Function evaluated at each unique minimizer
   * - number_excessive_descents
     - integer
     - Number of minimizers :math:`x_j^{(K_j)}` removed from :math:`T` when   
       finding unique minimizers.
