# METOD (Multistart With Early Termination of Descents)-Algorithm-
[![CI](https://github.com/Megscammell/METOD-Algorithm/actions/workflows/config.yml/badge.svg)](https://github.com/Megscammell/METOD-Algorithm/actions/workflows/config.yml)
[![codecov](https://codecov.io/gh/Megscammell/METOD-Algorithm/branch/master/graph/badge.svg?token=0HRI53L6BI)](https://codecov.io/gh/Megscammell/METOD-Algorithm)
[![Documentation Status](https://readthedocs.org/projects/metod-algorithm/badge/?version=latest)](https://metod-algorithm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/234310599.svg)](https://zenodo.org/badge/latestdoi/234310599)

Multistart is a global optimization technique, and works by applying local descent to a number of starting points. Multistart can be inefficient since local descent is applied to each starting point and the same local minimizers are discovered. For objective functions with locally quadratic behaviour close to the neighbourhoods of local minimizers, the METOD (Multistart with Early Termination of Descents) Algorithm can terminate many local descents early, which can greatly improve efficiency.

The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer, and often violates when the trajectories belong to different regions of attraction.


## Documentation
Documentation for the METOD-Algorithm can be found at https://metod-algorithm.readthedocs.io/.


## Installation
To install and test the METOD Algorithm, type the following into the command line:

```console
$ git clone https://github.com/Megscammell/METOD-Algorithm.git
$ cd METOD-Algorithm
$ python setup.py develop
$ pytest
```

## Quickstart
Apply the METOD Algorithm with an objective function and gradient.

```python
>>> import numpy as np
>>> import math
>>> import metod_alg as mt
>>>
>>> np.random.seed(90)
>>> d = 2
>>> A = np.array([[1, 0],[0, 10]])
>>> theta = np.random.uniform(0, 2 * math.pi)
>>> rotation = np.array([[math.cos(theta), -math.sin(theta)],
...                     [math.sin(theta), math.cos(theta)]])
>>> x0 = np.array([0.5, 0.2])
>>>
>>> def f(x, x0, A, rotation):
...     return 0.5 * (x - x0).T @ rotation.T @ A @ rotation @ (x - x0)
...
>>> def g(x, x0, A, rotation):
...     return rotation.T @ A @ rotation @ (x - x0)
...
>>> args = (x0, A, rotation)
>>> (discovered_minimizers,
...  number_minimizers,
...  func_vals_of_minimizers,
...  excessive_no_descents, 
...  starting_points,
...  no_grad_evals) = mt.metod(f, g, args, d, num_points=10)
>>> assert(np.all(np.round(discovered_minimizers[0], 3) == np.array([0.500, 0.200])))
>>> assert(number_minimizers == 1)
>>> assert(np.round(func_vals_of_minimizers, 3) == 0)
>>> assert(excessive_no_descents == 0)
>>> assert(np.array(starting_points).shape == (10, d))

```

## Examples

Examples of the METOD Algorithm applied with two different objective functions are available as Jupyter notebooks and Python scripts. All examples can be found at https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples. All examples have an intuitive layout and structure, which can be easily followed. 
