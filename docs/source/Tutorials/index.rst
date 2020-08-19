Tutorials
===================================

We will walk through two examples using the METOD Algorithm. 

Example 1
----------
.. _ex1:

Consider the minimum of several quadratic forms objective function

.. math::
   f(x)=\min_{1\le p \le P} \frac{1}{2}  (x-x_{0p})^T A_p^T \Sigma_p A_p (x-x_{0p}),

where :math:`P` is the number of minima; :math:`A_p` is a random rotation
matrix of size :math:`d\times d`; :math:`\Sigma_p` is a diagonal positive
definite matrix of size :math:`d\times d` with smallest and largest
eigenvalues :math:`\lambda_{min}` and :math:`\lambda_{max}` respectively;
:math:`x_{0p} \in \mathfrak{X}` and :math:`p=1,...,P`.

To run the METOD Algorithm in Python for the minimum of several quadratic 
forms objective function, navigate to METOD-Algorithm/Examples/Python.

The metod_quad_example.py program contains an overview of each of the steps to 
update in order to run the METOD algorithm using the minimum of several 
quadratic forms objective function. 

Each required parameter for the METOD algorithm within metod_quad_example.py 
has been set to the following: :math:`P = 5`; :math:`d = 50`; :math:`\lambda_
{min} = 1` and :math:`\lambda_{max} = 10`.
Therefore, we are able to run the program without making any changes to the
code.

To run the program, type ::

   $ python metod_quad_example.py

There will be no printed outputs, as all outputs are saved within two csv 
files. If metod_quad_example.py is run without making any changes to the code, 
the following csv files will be stored in the same directory as 
metod_quad_example.py

* **func_vals_discovered_minimas_d_50_p_5_quad.csv** : Function values of each of the discovered local minima.

* **summary_table_d_50_p_5_quad.csv** : Summary table containing the total number of unique local minima and extra descents.


Example 2
----------
.. _ex2:

Consider the Sum of Gaussians objective function

.. math::
   f(x)= -\sum_{p=1}^{P} c_p\exp \Bigg\{ {-\frac{1}{2 \sigma^2}(x-x_{0p})^T A_p^T \Sigma_p A_p(x-x_{0p})}\Bigg\}\,

where :math:`P` is the number of Gaussian densities; :math:`A_p` is a randomly 
chosen rotation matrices of size :math:`d\times d`; :math:`\Sigma_p` is a 
diagonal positive definite matrices of size :math:`d\times d` with smallest 
and largest eigenvalues :math:`\lambda_{min}` and :math:`\lambda_{max}
`respectively;  :math:`x_{0p} \in \mathfrak{X}` (centers of the Gaussian 
densities); :math:`c_p` is a fixed constants and :math:`p=1,...,P`.

Similarly to Example 1, to run the METOD Algorithm in Python for the Sum of
Gaussians objective function, navigate to METOD-Algorithm/Examples/Python.

The metod_sog_example.py program contains an overview of each of the steps to
update in order to run the METOD algorithm using the Sum of Gaussians
objective function.

Each required parameter for the METOD algorithm within metod_sog_example.py 
has been set to the following: :math:`P = 5`; :math:`d = 100`; :math:`\lambda_
{min} = 1`, :math:`\lambda_{max} = 10` and :math:`\sigma^2 = 4`.
Therefore, we are able to run the program without making any changes to the
code.

To run the program, type ::

   $ python metod_sog_example.py

There will be no printed outputs, as all outputs are saved within two csv 
files. If metod_sog_example.py is run without making any changes to the code, 
the following csv files will be stored in the same directory as 
metod_quad_example.py

* **func_vals_discovered_minimas_d_100_p_5_sog.csv** : Function values of each of the discovered local minima.

* **summary_table_d_100_p_5_sog.csv** : Summary table containing the total number of unique local minima and extra descents.


Jupyter Notebooks
-----------------

We can run :ref:`Example 1<ex1>` and :ref:`Example 2<ex2>` by using Jupyter notebooks:

* METOD Algorithm - Minimum of several quadratic forms.ipynb
* METOD Algorithm - Sum of Gaussians.ipynb

Notebooks can be found in METOD-Algorithm/Examples/Notebooks. Each Jupyter Notebook contains inbuilt instructions on how to update parameters and how to run the METOD Algorithm. Similar to :ref:`Example 1<ex1>` and :ref:`Example 2<ex2>`, outputs will be stored within two csv files. 
