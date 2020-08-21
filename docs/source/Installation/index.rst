.. _installation:

Installation
=============
To install the METOD Algorithm from source: ::

   $ git clone https://github.com/Megscammell/METOD-Algorithm.git
   $ cd METOD-Algorithm
   $ python setup.py install

To ensure all tests are working, create an environment and run the tests using pytest: ::

   $ conda env create -f environment.yml
   $ conda activate metod_algorithm
   $ pytest
