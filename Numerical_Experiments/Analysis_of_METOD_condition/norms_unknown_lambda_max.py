import numpy as np
import sys
from time import process_time
from time import perf_counter
import matplotlib.pyplot as plt

from metod_alg import metod_analysis as mt_ays
from metod_alg import objective_functions as mt_obj


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
        bound_1 = -5
        bound_2 = 5

    
    elif func_name == 'qing':
        d = 5
        g = mt_obj.qing_gradient
        func_args = (d,)
        bound_1 = -3
        bound_2 = 3


    elif func_name == 'zak':
        d = 10
        g = mt_obj.zakharov_grad
        func_args = (d,)
        bound_1 = -5
        bound_2 = 10


    elif func_name == 'hart':
        d = 6
        g = mt_obj.hartmann6_grad
        a, c, p = mt_obj.hartmann6_func_params()
        func_args = d, a, c, p
        bound_1 = 0
        bound_2 = 1
    return d, g, func_args, bound_1, bound_2


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def boxplots(store_all_norm_of_grad, num_functions):
    """
    Boxplots for approximation of beta.

    Parameters
    ----------
    store_all_norm_of_grad : 3-D array with shape (len(func_name_list),
                             num_functions, num_points).
                            The norm of the gradient at each starting point
                            for each function.
    num_functions : integer
                    Number of functions.
    """
    store_upd = np.zeros((3, num_functions))
    for j in range(1,4):
        for k in range(num_functions):
            store_upd[j - 1, k] = 2 / np.mean(store_all_norm_of_grad[j, k])
    
    plt.figure(figsize=(5, 5))
    bp = plt.boxplot(store_upd.T)
    set_box_color(bp, 'navy')
    plt.xticks(np.arange(1, 4), ['Styblinski-Tang', 'Qing', 'Zakharov'])
    plt.ylabel(r'Approximation of $\beta$')
    plt.yscale(value='log')
    plt.savefig('Approximation_beta_METOD.png')


if __name__ == "__main__":
    func_name_list = ['hart', 'styb', 'qing', 'zak'] 
    
    num_functions = 100
    num_points = 30
    avg_norm_of_grad = np.zeros((len(func_name_list)))
    store_all_norm_of_grad = np.zeros((len(func_name_list), num_functions,
                                       num_points))
    index = 0
    for func_name in func_name_list:
        d, g, func_args, bound_1, bound_2 = func_parameters(func_name)
        for j in range(num_functions):
            for i in range(num_points):
                x = np.random.uniform(bound_1, bound_2, (d, ))
                store_all_norm_of_grad[index, j, i] = np.linalg.norm(g(x, *func_args))
        avg_norm_of_grad[index] = np.mean(store_all_norm_of_grad[index])
        index += 1
    np.savetxt('avg_norm_grad.csv',
                 avg_norm_of_grad, delimiter=",")
    boxplots(store_all_norm_of_grad, num_functions)
