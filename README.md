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
Apply ```METOD``` with an objective function and gradient.

```python
import numpy as np
import math
import metod as mt


def f(x, A, rotation, x0):
    """
    Quadratic function used to test the METOD algorithm.

    Parameters
    ----------
        x :  1-D array with shape (d, ), where d is the dimension
        A : symmetric matrix with shape (d, d)
        rotation : rotation matrix with shape (d, d)
        x0 : 1-D array with shape (d, )

    Note
    -----
    To apply METOD, the function must have the form

    `f(x, *args) -> float`

    """
    return 0.5 * (x - x0).T @ rotation.T @ A @ rotation @ (x - x0)
    
def g(x, A, rotation, x0):
    """
    Quadratic gradient used to test the METOD algorithm.

    Parameters
    ----------
        x :  1-D array with shape (d, ), where d is the dimension
        A : symmetric matrix with shape (d, d)
        rotation : rotation matrix with shape (d, d)
        x0 : 1-D array with shape (d, )


    Note
    -----
    To apply METOD, the gradient must have the form

    g(x, *args) -> 1-D array with shape (d, )`

    """
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
assert(excessive_no_descents == 0)```


## Examples

The available METOD algorithm examples are applied with two objective functions. The first objective function is the minimum of several quadratic forms


![equation](<a href="https://www.codecogs.com/eqnedit.php?latex=f(x_n^{(k)})=\min_{1\le&space;p&space;\le&space;P}&space;(x_n^{(k)}-x_{0p})^T&space;A_p^T&space;\Sigma_p&space;A_p&space;(x_n^{(k)}-x_{0p})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x_n^{(k)})=\min_{1\le&space;p&space;\le&space;P}&space;(x_n^{(k)}-x_{0p})^T&space;A_p^T&space;\Sigma_p&space;A_p&space;(x_n^{(k)}-x_{0p})" title="f(x_n^{(k)})=\min_{1\le p \le P} (x_n^{(k)}-x_{0p})^T A_p^T \Sigma_p A_p (x_n^{(k)}-x_{0p})" /></a>)

where *P* is the number of minima; *A_p* is a randomly chosen rotation matrix of size *d\times d*; *\Sigma_p* is a diagonal positive definite matrix of size *d\times d* with smallest and largest eigenvalues *\lambda_{min}* and *\lambda_{max}* respectively;  *x_{0p} \in \mathfrak{X}* and *p=1,...,P*.
 

The second objective function is the Sum of Gaussians,

![equation](<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=&space;-\sum_{p=1}^{P}&space;c_p\exp&space;\Bigg\{&space;{-\frac{1}{2&space;\sigma^2}(x-x_{0p})^T&space;A_p^T&space;\Sigma_p&space;A_p(x-x_{0p})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=&space;-\sum_{p=1}^{P}&space;c_p\exp&space;\Bigg\{&space;{-\frac{1}{2&space;\sigma^2}(x-x_{0p})^T&space;A_p^T&space;\Sigma_p&space;A_p(x-x_{0p})}" title="f(x)= -\sum_{p=1}^{P} c_p\exp \Bigg\{ {-\frac{1}{2 \sigma^2}(x-x_{0p})^T A_p^T \Sigma_p A_p(x-x_{0p})}" /></a>)



