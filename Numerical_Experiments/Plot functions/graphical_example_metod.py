import numpy as np
import sys
import matplotlib.pyplot as plt

from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def calculate_distances(store_points, store_partner_points):
    """
    Compute the distances between two pairs of points. That is
    compute || x_1^{(k)} - x_2^{(k)} || and
    || \tilde{x}_1^{(k)} - \tilde{x}_2^{(k)} ||, where k is the iteration
    number.

    Parameters
    ----------
    store_points: list
                  Contains trajectories x_1^{(k)} and x_2^{(k)}.
    store_partner_points: list
                          Contains trajectories of partner points
                          \tilde{x}_1^{(k)} and \tilde{x}_2^{(k)}.

    Returns
    -------
    store_points_dist : 1-D array
                        Distances || x_1^{(k)} - x_2^{(k)} ||.
    store_partner_points_dist : 1-D array
                                Distances || \tilde{x}_1^{(k)} -
                                \tilde{x}_2^{(k)} ||.
    """
    store_points_dist = np.zeros((len(store_points[0])))
    store_partner_points_dist = np.zeros((len(store_points[0])))
    for j in range(len(store_points[0])):
        store_points_dist[j] = np.linalg.norm(store_points[0][j] -
                                              store_points[1][j])
        store_partner_points_dist[j] = (np.linalg.norm(
                                        store_partner_points[0][j] -
                                        store_partner_points[1][j]))
    return store_points_dist, store_partner_points_dist


def produce_contour_plot(seed, test_num, met, plot_type):
    """
    Generate contour plot for the minimum of several quadratic forms
    function. Also plot steepest descent iterations from two points
    which either belong to the same region of attraction or different regions
    of attraction.

    Parameters
    ----------
    seed : integer
           Seed used to initialize the pseudo random number generator.
    test_num : integer
               Number of points to evaluate function to compute contour plot.
    met : string
          Method to compute step length for steepest descent iterations.
    plot_type : string
                If plot_type = 'same', then two starting points will be from
                the same region of attraction.
                Otherwise, if plot_type = 'diff', then two starting points
                will be from different regions of attraction.
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

    x = np.linspace(0, 1, test_num)
    y = np.linspace(0, 1, test_num)
    Z = np.zeros((test_num, test_num))
    X, Y = np.meshgrid(x, y)
    for i in range(test_num):
        for j in range(test_num):
            x1_var = X[i, j]
            x2_var = Y[i, j]
            Z[i, j] = f(np.array([x1_var, x2_var]).reshape(2, ), *args)
    plt.figure(figsize=(5, 5))
    relax_sd_it = 1
    usage = 'metod_algorithm'
    tolerance = 0.00001
    projection = False
    bound_1 = 0
    bound_2 = 1
    option = 'minimize'
    initial_guess = 0.05
    num_p = 2
    beta = 0.1
    if plot_type == 'same':
        store_x = np.array([[0.5, 0.55],
                            [0.5, 0.9]])
    if plot_type == 'diff':
        store_x = np.array([[0.5, 0.55],
                            [0.7, 0.4]])
    store_points = []
    store_partner_points = []
    for k in range(num_p):
        x = store_x[k]
        (descended_x_points, its,
         grads) = (mt_alg.apply_sd_until_stopping_criteria
                   (x, d, projection, tolerance, option,
                    met, initial_guess, args, f, g,
                    bound_1, bound_2, usage, relax_sd_it,
                    None))
        store_points.append(descended_x_points[:5])
        sd_partner_points = mt_alg.partner_point_each_sd(descended_x_points,
                                                         beta, grads)
        store_partner_points.append(sd_partner_points[:5])

        chosen_x1 = descended_x_points[0:descended_x_points.shape[0]][:, 0]
        chosen_x2 = descended_x_points[0:descended_x_points.shape[0]][:, 1]

        chosen_z1 = sd_partner_points[0:sd_partner_points.shape[0]][:, 0]
        chosen_z2 = sd_partner_points[0:sd_partner_points.shape[0]][:, 1]

        plt.scatter(chosen_x1[0], chosen_x2[0], s=80, color='blue', marker='o')
        plt.scatter(chosen_x1[1:5], chosen_x2[1:5], s=20, color='blue')
        plt.scatter(chosen_z1[:5], chosen_z2[:5], s=20, color='red')
        plt.plot(chosen_x1[:5], chosen_x2[:5], 'blue')
    plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.savefig('anti_grad_its_quad_d=2_rs_%s_%s.png' % (seed, plot_type))
    (store_points_dist,
     store_partner_points_dist) = calculate_distances(store_points,
                                                      store_partner_points)

    np.savetxt('store_dist_points_quad_d=2_rs_%s_%s.csv' %
               (seed, plot_type),
               np.round(store_points_dist, 3),
               delimiter=',')

    np.savetxt('store_dist_partner_points_quad__d=2_rs_%s_%s.csv' %
               (seed, plot_type),
               np.round(store_partner_points_dist, 3),
               delimiter=',')


if __name__ == "__main__":
    plot_type = str(sys.argv[1])
    test_num = 100
    seed = 5
    met = 'Nelder-Mead'
    produce_contour_plot(seed, test_num, met, plot_type)
