.. role:: bash(code)
   :language: bash

Tutorials
===================================

We will walk through two examples using the METOD Algorithm. 

.. _ex1:

Example 1
-----------

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

In order to run metod_quad_example.py, values for :bash:`d`, :bash:`seed`, :bash:`P`, :bash:`lambda_1` and :bash:`lambda_2` will need to be provided.
For example, to set :bash:`d=50`, :bash:`seed=90`, :bash:`P=5`, :bash:`lambda_1=1` and :bash:`lambda_2=10`, type the following into the command line. ::

   $ python metod_quad_example.py 50 90 5 1 10

All outputs are saved within three csv 
files and information on all csv files can be found in the table below. All csv files will be stored in the same directory as 
metod_quad_example.py.

.. list-table::
   :widths: 33 30
   :header-rows: 1

   * - File name
     - Description
   * - discovered_minimizers_d_50_p_5_quad.csv
     - All local minimizers found by applying the METOD Algorithm.
   * - func_vals_discovered_minimizers_d_50_p_5_quad.csv
     - Function values at each discovered local minimizer.
   * - summary_table_d_50_p_5_quad.csv
     - Summary table containing the total number of unique local minimizers and repeated local descents to the same local minimizer.


.. _ex2:

Example 2
-----------

Consider the Sum of Gaussians objective function

.. math::
   f(x)= -\sum_{p=1}^{P} c_p\exp \Bigg\{ {-\frac{1}{2 \sigma^2}(x-x_{0p})^T A_p^T \Sigma_p A_p(x-x_{0p})}\Bigg\}\,

where :math:`P` is the number of Gaussian densities; :math:`A_p` is a random
rotation matrix of size :math:`d\times d`; :math:`\Sigma_p` is a 
diagonal positive definite matrix of size :math:`d\times d` with smallest 
and largest eigenvalues :math:`\lambda_{min}` and :math:`\lambda_{max}
`respectively;  :math:`x_{0p} \in \mathfrak{X}`; :math:`c_p` is a fixed constant and :math:`p=1,...,P`.

To run the METOD Algorithm in Python for the Sum of
Gaussians objective function, navigate to the `Python examples folder <https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples/Python>`_.

In order to run metod_sog_example.py, values for :bash:`d`, :bash:`seed`, :bash:`P`, :bash:`sigma_sq`, :bash:`lambda_1` and :bash:`lambda_2` will need to be provided.
For example, to set :bash:`d=20`, :bash:`seed=90`, :bash:`P=10`, :bash:`sigma_sq=0.8`, :bash:`lambda_1=1` and :bash:`lambda_2=10`, type the following into the command line. ::

   $ python metod_sog_example.py 20 90 10 0.8 1 10


All outputs are saved within three csv 
files and information on all csv files can be found in the table below. All csv files will be stored in the same directory as 
metod_sog_example.py

.. list-table::
   :widths: 33 30
   :header-rows: 1

   * - File name
     - Description
   * - discovered_minimizers_d_20_p_10_sog.csv
     - All local minimizers found by applying the METOD Algorithm.
   * - func_vals_discovered_minimizers_d_20_p_10_sog.csv
     - Function values at each discovered local minimizer.
   * - summary_table_d_20_p_10_sog.csv
     - Summary table containing the total number of unique local minimizers and repeated local descents to the same local minimizer.


.. _notebooks:

Jupyter Notebooks
----------------------

:ref:`Example 1 <ex1>` and :ref:`Example 2 <ex2>` are also in the form of Jupyter Notebooks:

* METOD Algorithm - Minimum of several quadratic forms.ipynb
* METOD Algorithm - Sum of Gaussians.ipynb

Notebooks can be found `here <https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples/Notebooks>`_.
Each Jupyter Notebook contains instructions on how to update parameters and how to run the METOD Algorithm.
Similar to :ref:`Example 1 <ex1>` and :ref:`Example 2 <ex2>`, outputs will be stored within csv files. 
