# METOD (Multistart With Early Termination of Descents)-Algorithm-

[![Build Status](https://app.travis-ci.com/Megscammell/METOD-Algorithm.svg?branch=master)](https://app.travis-ci.com/Megscammell/METOD-Algorithm)
[![Coverage Status](https://coveralls.io/repos/github/Megscammell/METOD-Algorithm/badge.svg?branch=master)](https://coveralls.io/github/Megscammell/METOD-Algorithm?branch=master)
[![Documentation Status](https://readthedocs.org/projects/metod-algorithm/badge/?version=latest)](https://metod-algorithm.readthedocs.io/en/latest/?badge=latest)

Multistart is a global optimization technique and works by applying local descent to a number of starting points. Multistart can be inefficient, as local descent is applied to each starting point and the same local minimizers are discovered. The METOD (Multistart with Early Termination of Descents) Algorithm can be more efficient than multistart, as some local descents are stopped early. This avoids repeated descents to the same local minimizer.

The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer, and often violates when the trajectories belong to different regions of attraction.


## Documentation
Documentation for the METOD-Algorithm can be found at https://megscammell.github.io/METOD-Algorithm/.


## Installation
To install the METOD Algorithm repository, type the following into the command line:

```console
$ git clone https://github.com/Megscammell/METOD-Algorithm.git
$ cd METOD-Algorithm
$ python setup.py develop
```

## Quickstart
Apply ```METOD``` with an objective function and gradient.

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
...  starting_points) = mt.metod(f, g, args, d, num_points=10)
>>> assert(np.all(np.round(discovered_minimizers[0], 3) == np.array([0.500, 0.200])))
>>> assert(number_minimizers == 1)
>>> assert(np.round(func_vals_of_minimizers, 3) == 0)
>>> assert(excessive_no_descents == 0)
>>> assert(np.array(starting_points).shape == (10, d))

```

## Examples

Examples of the METOD Algorithm applied with two different objective functions are available as Jupyter notebooks and Python scripts. All examples can be found at https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples. All examples have an intuitive layout and structure, which can be easily followed. 
