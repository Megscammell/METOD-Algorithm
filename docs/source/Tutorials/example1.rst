.. role:: bash(code)
   :language: bash

.. _ex1:

Example 1
=============

Consider the minimum of several quadratic forms objective function

.. math::
   f(x)=\min_{1\le p \le P} \frac{1}{2}  (x-x_{0p})^T A_p^T \Sigma_p A_p (x-x_{0p}),

where :math:`P` is the number of minima; :math:`A_p` is a random rotation
matrix of size :math:`d\times d`; :math:`\Sigma_p` is a diagonal positive
definite matrix of size :math:`d\times d` with smallest and largest
eigenvalues :math:`\lambda_{min}` and :math:`\lambda_{max}` respectively;
:math:`x_{0p} \in \mathfrak{X}` and :math:`p=1,...,P`.

To run the METOD Algorithm in Python for the minimum of several quadratic 
forms objective function, navigate to the `Python examples folder <https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples/Python>`_.

The metod_quad_example.py program contains an example on how to use the METOD algorithm. 

In order to run metod_quad_example.py, values for :bash:`d`, :bash:`seed`, :bash:`P`, :bash:`lambda_1` and :bash:`lambda_2` will need to be provided.
For example, if we wish to set :bash:`d=50`, :bash:`seed=90`, :bash:`P=5`, :bash:`lambda_1=1` and :bash:`lambda_2=10`, we would type the following into the command line::

   $ python metod_quad_example.py 50 90 5 1 10

There will be no printed outputs, as all outputs are saved within three csv 
files. If the above code is executed in the command line, the following csv files will be stored in the same directory as 
metod_quad_example.py

* **discovered_minimizers_d_50_p_5_quad.csv** : All local minimizers found by applying the METOD algorithm.

* **func_vals_discovered_minimizers_d_50_p_5_quad.csv** : Function values of each discovered local minimizer.

* **summary_table_d_50_p_5_quad.csv** : Summary table containing the total number of unique local minimizers and extra descents.
