import numpy as np
import sys
import scipy
from scipy import optimize
import matplotlib.pyplot as plt


def f(x):
    """
    Compute function value
    """
    return 4*x**2 + x**3


def new_f(x, coeff1, coeff2, coeff3):
    """
    Estimate parabola.
    """
    return coeff1*x**2 + coeff2*x + coeff3


def plot_graphs(x1, x2, x3, new_p, fx1, fx2, fx3, fnew_p, type):
    """
    Either plot functions f(x) and new_f(x, coeff1, coeff2, coeff3) or plot
    f(x) along with the point in which the gradient of the parabola is zero.
    """
    test_num = 100
    plt.clf()
    x = np.linspace(-2, 1.2, test_num)
    plt.xlim(-2, 1.2)
    y = f(x)
    plt.plot(x, y, color='black')
    plt.scatter(x1, fx1, color='green', marker='o', s=80)
    plt.scatter(x2, fx2, color='green', marker='o', s=80)
    plt.scatter(x3, fx3, color='green', marker='o', s=80)
    plt.annotate('$x_1$', (0, 0), (-1.5, 6), size=20)
    plt.annotate('$x_2$', (0, 0), (-0.6, 1.5), size=20)
    plt.annotate('$x_3$', (0, 0), (0.8, 5.6), size=20)
    if type == 'graphs':
        y_new = new_f(x, *coeffs)
        plt.plot(x, y_new, color='red')
        plt.savefig('spi_ex_plot_curve.png')
    else:
        plt.scatter(new_p, fnew_p, color='red', marker='o', s=80)
        plt.annotate('$x_4$', (0, 0), (-0.3, 0.6), size=20)
        plt.savefig('spi_ex_plot_points.png')


def plot_bracket_graphs(a, b, c, fa, fb, fc):
    test_num = 100
    plt.clf()
    x = np.linspace(-2, 1.2, test_num)
    plt.xlim(-2, 1.2)
    y = f(x)
    plt.plot(x, y, color='black')
    plt.scatter(a, fa, color='green', marker='o', s=80)
    plt.scatter(b, fb, color='green', marker='o', s=80)
    plt.annotate('$a$', (0, 0), (-1.5, 6), size=20)
    plt.annotate('$b$', (0, 0), (-0.6, 1.5), size=20)
    if c is not None and fc is not None:
        plt.scatter(c, fc, color='green', marker='o', s=80)
        plt.annotate('$c$', (0, 0), (0.9, 6.5), size=20)
    plt.savefig('bracket_ex_plot_points_%s.png' % (c is not None))


if __name__ == "__main__":
    type_ex = str(sys.argv[1])
    if type_ex == 'SPI':
        x = np.array([-1.5, 1, -0.5])
        y = f(x)
        coeffs = np.polyfit(x, y, 2)
        new_p = -coeffs[1] / (2 * coeffs[0])
        fnew_p = f(new_p)
        plot_graphs(x[0], x[1], x[2], new_p, y[0], y[1], y[2],
                    fnew_p, 'graphs')
        plot_graphs(x[0], x[1], x[2], new_p, y[0], y[1], y[2],
                    fnew_p, 'points')
    elif type_ex == 'bracket':
        a = -1.5
        fa = f(a)
        b = -0.5
        fb = f(b)
        (a, b, c,
         fa, fb, fc,
         func_calls) = scipy.optimize.bracket(f, a, b, args=())
        plot_bracket_graphs(a, b, None, fa, fb, None)
        plot_bracket_graphs(a, b, c, fa, fb, fc)
