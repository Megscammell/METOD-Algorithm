.. role:: bash(code)
   :language: bash

Required Inputs
===================

The required inputs of :bash:`metod.py` are listed below, along with the variable type. All required inputs need to be updated before running the METOD algorithm. 


.. list-table::
   :widths: 10 10 50
   :header-rows: 1

   * - Input parameter
     - Type
     - Description
   * - :bash:`f`
     - function
     - Function evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a float.
   * - :bash:`g`
     - function
     - Gradient evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a 1-D array with shape :bash:`(d, )`.
   * - :bash:`func_args`
     - tuple
     - Any function parameters needed to compute :bash:`f` or :bash:`g`.
   * - :bash:`d`
     - integer
     - Size of dimension.






