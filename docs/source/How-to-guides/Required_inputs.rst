.. role:: bash(code)
   :language: bash

Required Inputs
===================

The required inputs of :bash:`metod.py` are listed below, along with the variable type. All required inputs need to be updated before running the METOD algorithm. 

.. _fg:

Function and gradient
----------------------

To run the METOD algorithm, it is vital that a function and gradient is defined.

* :bash:`f` : Function evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a float.

* :bash:`g` : Gradient evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a 1-D array with shape :bash:`(d, )`.

* func_args : Any function parameters needed to compute :bash:`f` or :bash:`g`.

Two different Python programs to compute objective functions and gradients can be found 
`here <https://github.com/Megscammell/METOD-Algorithm/tree/master/src/metod/objective_functions>`_. The programs are called :bash:`quad_function.
py` and :bash:`quad_gradient.py` for the minimum of several quadratic forms objective 
function and :bash:`sog_function.py` and :bash:`sog_gradient.py` for the Sum of Gaussians objective function.

If a different objective function and gradient is required, it is important 
that they have the same form as described for :bash:`f` and :bash:`g`. Python programs to generate random function parameters for both the minimum of several quadratic forms and Sum of Gaussians objective functions can be found `here <https://github.com/Megscammell/METOD-Algorithm/tree/master/src/metod/objective_functions>`_. The programs are called :bash:`function_parameters_quad.py`
and :bash:`function_parameters_sog.py` respectively and are used for :bash:`func_args`. If a 
different function and gradient is used, function parameters for the new 
function and gradient will also need to be generated.

.. _d:

Dimension
------------

Input parameter, :bash:`d`, is the size of dimension. The METOD algorithm works well for high dimensions (i.e :bash:`d=100`).









