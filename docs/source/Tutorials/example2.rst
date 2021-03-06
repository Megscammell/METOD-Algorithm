.. role:: bash(code)
   :language: bash

.. _ex2:

Example 2
===================================

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

The metod_sog_example.py program contains an example on how to use the METOD algorithm. 

In order to run metod_sog_example.py, values for :bash:`d`, :bash:`seed`, :bash:`P`, :bash:`sigma_sq`, :bash:`lambda_1` and :bash:`lambda_2` will need to be provided.
For example, if we wish to set :bash:`d=20`, :bash:`seed=90`, :bash:`P=10`, :bash:`sigma_sq=0.8`, :bash:`lambda_1=1` and :bash:`lambda_2=10`, we would type the following into the command line.

To run the program, type the following into the command line ::

   $ python metod_sog_example.py 20 90 10 0.8 1 10


There will be no printed outputs, as all outputs are saved within three csv 
files. If the above code is executed in the command line, the following csv files will be stored in the same directory as 
metod_sog_example.py

* **discovered_minimizers_d_20_p_10_sog.csv** : All local minimizers found by applying the METOD algorithm.

* **func_vals_discovered_minimizers_d_20_p_10_sog.csv** : Function values at each discovered local minimizer.

* **summary_table_d_20_p_10_sog.csv** : Summary table containing the total number of unique local minimizers and extra descents.
