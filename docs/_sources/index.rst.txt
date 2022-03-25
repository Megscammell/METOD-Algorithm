Welcome to METOD Algorithm's documentation!
===========================================
.. image:: https://github.com/Megscammell/METOD-Algorithm/actions/workflows/config.yml/badge.svg
   :target: https://github.com/Megscammell/METOD-Algorithm/actions/workflows/config.yml


.. image:: https://codecov.io/gh/Megscammell/METOD-Algorithm/branch/master/graph/badge.svg?token=0HRI53L6BI
   :target: https://codecov.io/gh/Megscammell/METOD-Algorithm


.. image:: https://readthedocs.org/projects/metod-algorithm/badge/?version=latest
   :target: https://metod-algorithm.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5112850.svg
   :target: https://doi.org/10.5281/zenodo.5112850


Multistart is a global optimization technique and works by applying local descent to a number of starting points. Multistart can be inefficient,
as local descent is applied to each starting point, and the same local minimizers are discovered. The Multistart with Early Termination of Descents (METOD)
Algorithm :cite:`metod_1` can be more efficient than Multistart since some local descents are stopped early, which reduces the number of repeated descents to the same local minimizer.

The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer,
and often violates when the trajectories belong to different regions of attraction.

All Python code for the METOD Algorithm can be found `here <https://github.com/Megscammell/METOD-Algorithm/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   background
   Installation/index
   Tutorials/index
   Inputs of the METOD Algorithm/index
   Outputs of the METOD Algorithm/index
   Usage/index
   References/index
   

 
   



