# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a global optimization technique and works by applying local descent to a number of starting points. Multistart can be inefficient, as local descent is applied to each starting point and the same local minimizers are discovered. The METOD (Multistart with Early Termination of Descents) algorithm can be more efficient than multistart, as some local descents are stopped early. This avoids repeated descents to the same local minimizer.

The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer, and often violates when the trajectories belong to different regions of attraction.

## Installation
To use the METOD Algorithm, please do the following:

1) Open the command line and navigate to where you would like the file to be stored and run the following code:
```python
git clone https://github.com/Megscammell/METOD-Algorithm.git
```
2) Navigate to the directory that contains the setup.py file and run the following code:
```python
python setup.py develop
```
3) To run tests, pytest is used which will be installed if step 2 has been completed successfully. In the same directory as step 3, run the following in the command line:
```python
pytest
```

## Quickstart
Apply ```METOD``` with an objective function and gradient.

```python
import numpy as np
import math
import metod as mt


def f(x, A, rotation, x0):
    """ Compute objective function."""
    return 0.5 * (x - x0).T @ rotation.T @ A @ rotation @ (x - x0)
    
def g(x, A, rotation, x0):
    """ Compute gradient of objective function."""
    return rotation.T @ A @ rotation @ (x - x0)

# Set up function and algorithm parameters.
d = 2
theta = np.random.uniform(0, 2 * math.pi)
rotation = np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
A = np.array([[1, 0], [0, 10]])
x0 = np.array([0.5, 0.2])
args = A, rotation, x0

# Run the METOD algorithm with optional input parameter num_points=10.
(discovered_minimizers,
 number_minimizers,
 func_vals_of_minimizers,
 excessive_no_descents)  = mt.metod(f, g, args, d, num_points=10)

# Assert that outputs are correct.
assert(np.all(np.round(discovered_minimizers[0], 3) == np.array([0.500,0.200])))
assert(number_minimizers == 1)
assert(np.round(func_vals_of_minimizers, 3) == 0)
assert(excessive_no_descents == 0)
```

## Examples

Examples of the METOD algorithm applied with two different objective functions are available as Jupyter notebooks and Python scripts. All examples can be found in https://github.com/Megscammell/METOD-Algorithm/tree/master/Examples. Jupyter notebook examples provide a user friendly interface, with details on running the METOD algorithm for two different objective functions. All examples have an intuitive layout and structure, which can be easily followed. 
