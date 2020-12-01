# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a celebrated global optimization technique, which applies steepest descent iterations to an initial starting point in order to find a local minimizer. The METOD algorithm can be more efficient than Multistart as some iterations of steepest descent are stopped early if certain conditions are satisfied. This avoids repeated descents to the same minimizer(s). 

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
To apply ```METOD``` with ```f(x, *args)```

```python
import numpy as np
import metod as mt

def f(x, A, x0):
"""
Quadratic function used to test the METOD algorithm.

Parameters
----------
    x :  a 1-D array with shape (d, )
    A : symmetric matrix
    x0 : local minima

Note
-----
To apply METOD, the function must have the form

`f(x, *args) -> float`

"""
    return 0.5 * (x - x0).T @ A @ (x - x0)
    
def g(x, A, x0):

"""
Quadratic gradient used to test the METOD algorithm.

Parameters
----------
    x :  a 1-D array with shape (d, )
    A : symmetric matrix
    x0 : local minima
    
    
Note
-----
To apply METOD, the gradient must have the form

g(x, *args) -> 1-D array with shape (d, )`

"""
    return A @ (x - x0)

d = 2
A = np.array([[1, 0], [0, 10]])
x0 = np.array([0.5, 0.2])
args = A, x0

# Call the METOD algorithm and change the optional input parameter to num_points=10.

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