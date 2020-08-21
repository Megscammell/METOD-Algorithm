Required Inputs
===================

The required inputs of metod.py are listed below, along with the variable type.

* :ref:`f <fg>`
* :ref:`g <fg>`
* :ref:`func_args <fg>`
* :ref:`d <d>`

All required inputs need to be updated before running the METOD algorithm. 

.. _fg:

Function and gradient
----------------------

To run the METOD algorithm, it is vital that a function and gradient is defined.

* f : Function evaluated at a point, which is a 1-D array with shape (d, ), and outputs a float.

* g : Gradient evaluated at a point, which is a 1-D array with shape (d, ), and outputs a 1-D array with shape (d, ).

* func_args : Any function parameters needed to compute f or g.

Two different objective functions and gradients are contained within 
METOD-Algorithm/src/metod/objective_functions. These are called quad_function.
py and quad_gradient.py for the minimum of several quadratic forms objective 
function and sog_function.py and sog_gradient.py for the Sum of Gaussians 
objective function.

If a different objective function and gradient is required, it is important 
that they have the same form as described for f and g. Also contained within 
METOD-Algorithm/src/metod/objective_functions are programs to generate random 
function parameters for both the minimum of several quadratic forms and Sum of 
Gaussians objective functions. These are called function_parameters_quad.py 
and function_parameters_sog.py respectively and are used for func_args. If a 
different function and gradient is used, function parameters for the new 
function and gradient will also need to be generated.

.. _d:

Dimension
------------

Input parameter, d, is the size of dimension. The METOD algorithm works well for high dimensions (i.e d=100).









