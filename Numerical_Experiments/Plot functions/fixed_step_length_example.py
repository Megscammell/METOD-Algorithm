import numpy as np
import sys
import matplotlib.pyplot as plt

from metod_alg import objective_functions as mt_obj


def compute_its(x, d, g, args, step_size):
    """
    Function to compute iterations of descent where the step length is a
    fixed positive constant.

    Parameters
    ----------
    x : 1-D array with shape (d, )
        Apply descent iterations to point.
    d : integer
        Size of dimension.
    g : gradient of objective function.

       `g(x, *args) -> 1-D array with shape (d, )`

    args : tuple
                Arguments passed to the gradient g.
    step_size : float
                Positive constant used as the step size to compute iterations
                of descent.

    Returns
    -------
    sd_iterations : 2-D array
                    Each iteration of descent is stored in each row of
                    sd_iterations.
    """
    sd_iterations = np.zeros((1, d))
    sd_iterations[0, :] = x.reshape(1, d)
    while np.linalg.norm(g(x, *args)) > 0.1:
        x = x - step_size * g(x, *args)
        sd_iterations = np.vstack([sd_iterations, x.reshape
                                  ((1, d))])
    return sd_iterations


def illustrate_importance_of_step(seed, test_num, step_type):
    """
    Generate contour plot for minimum of several quadratic forms function,
    along with plot of descent iterations from a starting point (0.5, 0.55),
    with fixed step length.

    Parameters
    ----------
    seed : integer
           Seed used to initialize the pseudo random number generator.
    test_num : integer
               Number of points to evaluate function to compute contour plot.
    step_type : string
                Either select step_type = 'long' or step_type = 'short'. If
                step_type = 'long', then step_size = 0.6. Otherwise, if
                step_type = 'short', then step_size = 0.1.
    """
    np.random.seed(seed)
    d = 2
    P = 4
    lambda_1 = 1
    lambda_2 = 5

    f = mt_obj.several_quad_function
    g = mt_obj.several_quad_gradient
    store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                 (P, d, lambda_1, lambda_2))
    store_x0 = np.array([[0.96, 0.09],
                         [0.86, 0.9],
                         [0.2, 0.98],
                         [0.12, 0.22]])
    args = P, store_x0, matrix_combined

    x = np.linspace(0, 1.2, test_num)
    y = np.linspace(0, 1.2, test_num)
    Z = np.zeros((test_num, test_num))
    X, Y = np.meshgrid(x, y)
    for i in range(test_num):
        for j in range(test_num):
            x1_var = X[i, j]
            x2_var = Y[i, j]
            Z[i, j] = f(np.array([x1_var, x2_var]).reshape(2, ), *args)

    x = np.array([0.5, 0.55])
    if step_type == 'long':
        step_size = 0.6
    elif step_type == 'short':
        step_size = 0.1
    descended_x_points = compute_its(x, d, g, args, step_size)

    chosen_x1 = descended_x_points[0:descended_x_points.shape[0]][:, 0]
    chosen_x2 = descended_x_points[0:descended_x_points.shape[0]][:, 1]

    if step_type == 'long':
        plt.scatter(chosen_x1[0], chosen_x2[0], s=80, color='green',
                    marker='o')
        plt.scatter(chosen_x1[1:4], chosen_x2[1:4], s=20, color='blue')
        plt.plot(chosen_x1[:5], chosen_x2[:5], 'blue')
        plt.gca().set_xlim(left=0, right=1.2)
        plt.gca().set_ylim(bottom=0)
        plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.5)
    elif step_type == 'short':
        plt.scatter(chosen_x1[0], chosen_x2[0], s=80, color='green',
                    marker='o')
        plt.scatter(chosen_x1[1:], chosen_x2[1:], s=20, color='blue')
        plt.plot(chosen_x1, chosen_x2, 'blue')
        plt.gca().set_xlim(left=0, right=1)
        plt.gca().set_ylim(bottom=0)
        plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.5)
    plt.savefig('fixed_step_size_d=2_rs_%s_%s.png' % (seed, step_type))


if __name__ == "__main__":
    step_type = str(sys.argv[1])
    test_num = 100
    seed = 5
    illustrate_importance_of_step(seed, test_num, step_type)
