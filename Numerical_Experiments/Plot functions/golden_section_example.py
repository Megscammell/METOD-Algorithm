import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Compute quadratic function."""
    return x ** 2


def compute_c(a, b):
    """Compute value c for golden section search."""
    c = b + ((a - b) / 1.618)
    return c, f(c)


def compute_d(a, b):
    """Compute value d for golden section search."""
    d = a + ((b - a) / 1.618)
    return d, f(d)


def compute_graphs(it_num, a, b, c, d, test_num, f):
    """
    Display various graphs illustrating the functionality of golden section
    search.
    """
    plt.clf()
    x = np.linspace(-2, 1, test_num)
    plt.ylim(0, 4)
    plt.xlim(-2, 1)
    y = f(x)
    plt.plot(x, y, color='black')
    plt.scatter(c, fc, color='green', marker='o', s=80)
    plt.scatter(d, fd, color='green', marker='o', s=80)
    if it_num == 0:
        lower = 0.05
        upper = -0.5
    elif it_num == 1:
        lower = 0.05
        upper = -0.5
    elif it_num == 2:
        lower = 0.05
        upper = -0.5
    elif it_num == 3:
        lower = 0.04
        upper = -0.5
    plt.annotate('$b$', (0, 0), (b - lower, upper), size=20)
    plt.annotate('$a$', (0, 0), (a - lower, upper), size=20)
    plt.annotate('$x_1$', (0, 0), (c - lower, upper), size=20)
    plt.annotate('$x_2$', (0, 0), (d - lower, upper), size=20)
    plt.vlines(b, 0, 4, color='black')
    plt.vlines(a, 0, 4, color='black')
    plt.vlines(c, 0, 4, color='black')
    plt.vlines(d, 0, 4, color='black')
    if fc > fd:
        plt.axvspan(a, c, color='r', alpha=0.5, lw=0)
    elif fc <= fd:
        plt.axvspan(d, b, color='r', alpha=0.5, lw=0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('golden_section_ex_%s.png' % (it_num))


if __name__ == "__main__":
    num = 4
    test_num = 100
    b = 1
    a = -2
    c, fc = compute_c(a, b)
    d, fd = compute_d(a, b)
    for it_num in range(num):
        compute_graphs(it_num, a, b, c, d, test_num, f)
        if fc < fd:
            b = d
            d = c
            fd = fc
            c, fc = compute_c(a, b)
        else:
            a = c
            c = d
            fc = fd
            d, fd = compute_d(a, b)
