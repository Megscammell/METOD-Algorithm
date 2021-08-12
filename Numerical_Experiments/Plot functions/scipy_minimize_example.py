import numpy as np
import sys
import scipy
from scipy import optimize
import matplotlib.pyplot as plt


def f(x):
    """
    Compute function value.
    """
    return np.sin(x)


def g(x):
    """
    Compute derivative of function.
    """
    return np.cos(x)


def test_func(gamma, x, f, g):
    """
    Find smallest gamma along the search direction g(x).
    """
    return f(x - gamma * g(x))


def plot_graphs(a, b, fa, fb, type_ex):
    """
    Plot points a and b, along with the function f(x).
    """
    test_num = 100
    plt.clf()
    x = np.linspace(-4, 8, test_num)
    plt.xlim(-4, 8)
    y = f(x)
    plt.plot(x, y, color='black')
    plt.scatter(a, fa, color='green', marker='o', s=80)
    plt.scatter(b, fb, color='green', marker='o', s=80)
    plt.annotate('$x_n$', (0, 0), (-3, 0.05), size=18)
    if type_ex == 'overshoot':
        plt.annotate(r'$x_{n+1}$', (0, 0), (3.95, -0.7), size=18)
    else:
        plt.annotate(r'$x_{n+1}$', (0, 0), (-2.4, -0.7), size=18)
    plt.savefig('scipy_minimize_ex_plot_%s.png' % (type_ex))


if __name__ == "__main__":
    type_ex = str(sys.argv[1])
    a = -3
    fa = f(a)
    if type_ex == 'overshoot':
        gamma = 5
    else:
        gamma = 0.005
    res = scipy.optimize.minimize(test_func, gamma, args=(a, f, g))
    b = a - res.x * g(a)
    fb = f(a - res.x * g(a))
    plot_graphs(a, b, fa, fb, type_ex)
