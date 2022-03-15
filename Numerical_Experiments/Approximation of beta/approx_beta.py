import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from metod_alg import objective_functions as mt_obj


def approx_beta(norm_grads, d):
    """
    Approximates the value of the partner step size beta.

    Parameters
    ----------
    norm_grads : 1-d array
                 Array containing the norm of the gradient for each
                 starting point generated.
    d : integer
        Size of dimension.

    Returns
    -------
    approx_beta : float
                  Approximation of beta.
    """
    return ((1 / np.mean(norm_grads)) *
            (((d - 1) * (d - 1)) / (2.25 * d * np.sqrt(d))))


def numerical_quad_exp(P, d_list, lambda_2, seed):
    """
    Investigate the approximation of beta for the minimum of several quadratic
    forms objective function with various P, dimension and lambda_2.

    Parameters
    ----------
    p : integer
        Number of local minima.
    d_list : list
             Contains various dimensions to test.
    lambda_2 : integer
               Largest eigenvalue of diagonal matrix.
    seed : integer
           Random seed used to initialize the pseudo-random number generator.

    Returns
    -------
    store_quantity : 2-D array
                     Approximation of beta for various P, dimension and
                     lambda_2.
    """
    g = mt_obj.several_quad_gradient
    lambda_1 = 1
    num = 100
    num_func = 100
    store_quantity = np.zeros((len(d_list), num_func))
    index = 0
    np.random.seed(seed)
    for d in d_list:
        for i in range(num_func):
            store_x0, store_A = (mt_obj.function_parameters_several_quad(
                                 P, d, lambda_1, lambda_2))
            args = P, store_x0, store_A
            store_norm_grad = np.zeros((num))
            for j in range(num):
                grad = g(np.random.uniform(0, 1, (d,)), *args)
                store_norm_grad[j] = np.linalg.norm(grad)
            store_quantity[index, i] = approx_beta(store_norm_grad, d)
        index += 1
    return store_quantity


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots(arr1, arr2, arr3, labels, ticks, P, seed):
    """
    Boxplots of the approximation of beta for different P, dimension, and
    lambda_2 for the minimum of several quadratic forms function.
    """
    plt.figure(figsize=(5, 6))
    plt.ylim(0, 1)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*3.0-0.6)
    bpc = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*3.0+0)
    bpr = plt.boxplot(arr3.T,
                      positions=np.array(range(len(arr3)))*3.0+0.6)
    set_box_color(bpl, 'green')
    set_box_color(bpc, 'purple')
    set_box_color(bpr, 'navy')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.plot([], c='navy', label=labels[2])
    plt.legend(bbox_to_anchor=(0.56, 1.025), loc='upper left',
            prop={'size': 15})
    plt.xlabel(r'$d$', size=20)
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('quad_P=%s_seed=%s.png' %
                (P, seed))


def func_parameters(func_name):
    """
    Generates function parameters for a particular function.

    Parameters
    ----------
    func_name : string
                Name of function.

    Returns
    --------
    d : integer
        Size of dimension.
    g : gradient of objective function.

       `g(x, *func_args) -> 1-D array with shape (d, )`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    func_args : tuple
                Arguments passed to f and g.
    bound_1 : integer
               Lower bound used to generate starting points.
    bound_2 : integer
               Upper bound used to generate starting points.

    """
    if func_name == 'styb':
        d = 5
        g = mt_obj.styblinski_tang_gradient
        func_args = ()
        bounds = (-5, 5)

    elif func_name == 'qing':
        d = 5
        g = mt_obj.qing_gradient
        func_args = (d,)
        bounds = (-3, 3)

    elif func_name == 'zak':
        d = 10
        g = mt_obj.zakharov_grad
        func_args = (d,)
        bounds = (-5, 10)
    return d, g, func_args, bounds


def compute_approx_unknown_lambda_max(g, args, bounds, d, num_func,
                                      num_points):
    """
    Investigate the approximation of beta for different functions
    (i.e. Styblinski-Tang, Qing and Zakharov).

    Parameters
    ----------
    g : gradient of objective function.

       `g(x, *func_args) -> 1-D array with shape (d, )`

        where `x` is a 1-D array with shape(d, ) and func_args is a
        tuple of arguments needed to compute the gradient.
    args : tuple
           Arguments passed to f and g.
    bounds : tuple
             Used to generate starting points.
    d : integer
        Size of dimension.
    num_func : integer
               Number of functions to generate.
    num_points : integer
                 Number of points to generate.

    Returns
    -------
    store_quantity : 2-D array
                     Approximation of beta for different functions.
    """
    store_quantity = np.zeros((num_func))
    for i in range(num_func):
        store_norm_grad = np.zeros((num_points))
        for j in range(num_points):
            grad = g(np.random.uniform(*bounds, (d,)), *args)
            store_norm_grad[j] = np.linalg.norm(grad)
        store_quantity[i] = approx_beta(store_norm_grad, d)
    return store_quantity


def boxplots_unknown_lambda_max(seed):
    """
    Boxplots for approximation of beta.

    Parameters
    ----------
    seed : integer
           Random seed used to initialize the pseudo-random number generator.
    """
    func_name_list = ['styb', 'qing', 'zak']
    num_func = 100
    num = 100
    store_quantities = np.zeros((len(func_name_list), num_func))
    index = 0
    np.random.seed(seed)
    for func_name in func_name_list:
        d, g, func_args, bounds = func_parameters(func_name)
        store_quantities[index] = (compute_approx_unknown_lambda_max(
                                   g, func_args, bounds, d, num_func, num))
        index += 1
    plt.figure(figsize=(5, 6))
    bp = plt.boxplot(store_quantities.T)
    set_box_color(bp, 'navy')
    plt.xticks(np.arange(1, 4), ['Styblinski-Tang', 'Qing', 'Zakharov'],
               size=14)
    plt.yscale(value='log')
    plt.yticks(fontsize=14)
    plt.savefig('Approximation_beta_METOD_%s.png' % (seed))


if __name__ == "__main__":
    P = int(sys.argv[1])
    seed = int(sys.argv[2])
    type_func = str(sys.argv[3])
    if type_func == 'known':
        lambda_list = [5, 10, 20]
        if P == 5:
            d_list = [5, 20, 100]
        elif P == 20:
            d_list = [20, 50, 100]
        elif P == 50:
            d_list = [50, 100, 200]
        store_all_quantity = np.zeros((len(lambda_list), len(d_list), 100))
        index = 0
        for lambda_2 in tqdm.tqdm(lambda_list):
            store_all_quantity[index] = numerical_quad_exp(P, d_list, lambda_2, seed)
            index += 1
        ticks = d_list
        labels = [r'$\lambda_{max} = 5$',
                  r'$\lambda_{max} = 10$',
                  r'$\lambda_{max} = 20$']
        create_boxplots(store_all_quantity[0],
                        store_all_quantity[1],
                        store_all_quantity[2],
                        labels, ticks, P, seed)
    elif type_func == 'unknown':
        boxplots_unknown_lambda_max(seed)
