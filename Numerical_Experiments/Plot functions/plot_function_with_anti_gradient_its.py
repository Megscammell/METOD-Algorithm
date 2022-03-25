import numpy as np
import sys
import matplotlib.pyplot as plt

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def plot_functions_with_anti_gradient_its(obj, test_num, seed, num_p, met):
    """
    Generate contour plot for either the minimum of several quadratic forms
    or sum of gaussians function. Also plot steepest descent iterations from
    a number of starting points.

    Parameters
    ----------
    obj : string
          If obj = 'quad', then the minimum of several quadratic forms contour
          plot is generated. Otherwise, if obj = 'sog', then the sum of
          gaussians contour plot is generated.
    test_num : integer
               Number of points to evaluate function to compute contour plot.
    seed : integer
           Seed used to initialize the pseudo random number generator.
    num_p : integer
            Number of starting points to apply steepest descent iterations.
    met : string
          Method to compute step length for steepest descent iterations.

    """
    np.random.seed(seed)
    d = 2
    P = 5
    lambda_1 = 1
    lambda_2 = 4
    if obj == 'quad':
        f = mt_obj.several_quad_function
        g = mt_obj.several_quad_gradient
        store_x0, matrix_combined = (mt_obj.function_parameters_several_quad
                                     (P, d, lambda_1, lambda_2))
        store_x0 = np.array([[0.96, 0.09],
                             [0.86, 0.9],
                             [0.2, 0.98],
                             [0.12, 0.22],
                             [0.5, 0.5]])
        args = P, store_x0, matrix_combined
        bounds = (0, 1)

    elif obj == 'sog':
        f = mt_obj.sog_function
        g = mt_obj.sog_gradient
        store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog
                                              (P, d, lambda_1, lambda_2))

        store_x0 = np.array([[0.96, 0.09],
                             [0.86, 0.9],
                             [0.2, 0.98],
                             [0.12, 0.22],
                             [0.5, 0.5]])
        store_c = np.array([0.8, 0.7, 0.9, 0.75, 0.6])
        sigma_sq = 0.05
        args = P, sigma_sq, store_x0, matrix_combined, store_c
        bounds = (0, 1)

    elif obj == 'styb':
        f = mt_obj.styblinski_tang_function
        g = mt_obj.styblinski_tang_gradient
        bounds = (-5, 5)
        args = ()

    elif obj == 'qing':
        f = mt_obj.qing_function
        g = mt_obj.qing_gradient
        args = (d, )
        bounds = (-3, 3)

    x = np.linspace(*bounds, test_num)
    y = np.linspace(*bounds, test_num)
    Z = np.zeros((test_num, test_num))
    X, Y = np.meshgrid(x, y)
    for i in range(test_num):
        for j in range(test_num):
            x1_var = X[i, j]
            x2_var = Y[i, j]
            Z[i, j] = f(np.array([x1_var, x2_var]).reshape(2, ), *args)

    relax_sd_it = 1
    usage = 'metod_algorithm'
    tolerance = 0.00001
    projection = False
    bound_1 = bounds[0]
    bound_2 = bounds[1]
    option = 'minimize_scalar'
    initial_guess = 0.005
    plt.figure(figsize=(5, 5))

    for j in range(num_p):
        x = np.random.uniform(*bounds, (d, ))
        (descended_x_points, its,
         grads) = (mt_alg.apply_sd_until_stopping_criteria
                   (x, d, projection, tolerance, option,
                    met, initial_guess, args, f, g,
                    bound_1, bound_2, usage, relax_sd_it,
                    None))

        chosen_x1 = descended_x_points[0:descended_x_points.shape[0]][:, 0]
        chosen_x2 = descended_x_points[0:descended_x_points.shape[0]][:, 1]
        plt.scatter(chosen_x1[0], chosen_x2[0], s=50, color='green')
        plt.scatter(chosen_x1[1:], chosen_x2[1:], s=20, color='blue')
        plt.plot(chosen_x1, chosen_x2, 'blue')

    if obj == 'qing':
        plt.contour(X, Y, Z, 120, cmap='RdGy', alpha=0.4)
    else:
        plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.4)
    plt.colorbar()
    plt.savefig('anti_grad_its_%s_d=2_rs_%s.png' % (obj, seed))


if __name__ == "__main__":
    obj = str(sys.argv[1])
    met = 'Brent'
    test_num = 100
    num_p = 10
    seed = 2000
    plot_functions_with_anti_gradient_its(obj, test_num, seed, num_p, met)
