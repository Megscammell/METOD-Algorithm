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


def const_approx_beta(d, c):
    """
    Compute constant.

    Parameters
    ----------
    c : float
        Value used within computatio
    d : integer
        Size of dimension.

    Returns
    -------
    value : float
            Value to multiply the reciprocal of the average norm of the
            gradient.
    """
    return ((d - 1) * (d - 1)) / (c * d * np.sqrt(d))


def compute_norm_grad(d, lambda_2, P):
    """
    Compute constant.

    Parameters
    ----------
    d : integer
        Size of dimension.
    lambda_2 : integer
               Largest eigenvalue.
    P : integer
        Number of local minima.

    Returns
    -------
    value : float
            Average norm of the gradient.
    """
    g = mt_obj.several_quad_gradient
    lambda_1 = 1
    num = 30
    store_norm_grad = np.zeros((num))
    store_x0, store_A = (mt_obj.function_parameters_several_quad(
                         P, d, lambda_1, lambda_2))
    args = P, store_x0, store_A
    for j in range(num):
        grad = g(np.random.uniform(0, 1, (d, )), *args)
        store_norm_grad[j] = np.linalg.norm(grad)
    return np.mean(store_norm_grad)


def approx_beta_num_exp(d, lambda_2, P):
    """
    Compute approx of beta.

    Parameters
    ----------
    d : integer
        Size of dimension.
    lambda_2 : integer
               Largest eigenvalue.
    P : integer
        Number of local minima.

    Returns
    -------
    value : float
            Approx beta.
    """
    g = mt_obj.several_quad_gradient
    lambda_1 = 1
    num = 30
    store_norm_grad = np.zeros((num))
    store_x0, store_A = (mt_obj.function_parameters_several_quad(
                         P, d, lambda_1, lambda_2))
    args = P, store_x0, store_A
    for j in range(num):
        grad = g(np.random.uniform(0, 1, (d, )), *args)
        store_norm_grad[j] = np.linalg.norm(grad)
    return approx_beta(store_norm_grad, d)


def diff_d_norm_num_exp(P, start_d, end_d):
    """
    Compute approx of beta.

    Parameters
    ----------
    d : integer
        Size of dimension.
    start_d : integer
              First dimension to test.
    end_d : integer
            Last dimension to test.

    Returns
    -------
    value : float
            Approx beta.
    """
    store_norms = np.zeros((end_d - start_d, 3))
    index = 0
    for d in tqdm.tqdm(np.arange(start_d, end_d)):
        test_num = np.zeros((3, 100))
        for i in range(100):
            test_num[0, i] = 1 / compute_norm_grad(int(d), 5, P)
            test_num[1, i] = 1 / compute_norm_grad(int(d), 10, P)
            test_num[2, i] = 1 / compute_norm_grad(int(d), 20, P)
        store_norms[index, 0] = np.mean(test_num[0])
        store_norms[index, 1] = np.mean(test_num[1])
        store_norms[index, 2] = np.mean(test_num[2])
        index += 1
    return store_norms

def diff_d_approx_beta_num_exp(P, start_d, end_d):
    """
    Compute approx of beta.

    Parameters
    ----------
    d : integer
        Size of dimension.
    start_d : integer
              First dimension to test.
    end_d : integer
            Last dimension to test.

    Returns
    -------
    value : float
            Approx beta.
    """
    store_norms = np.zeros((end_d - start_d, 3))
    index = 0
    for d in tqdm.tqdm(np.arange(start_d, end_d)):
        test_num = np.zeros((3, 100))
        for i in range(100):
            test_num[0, i] = approx_beta_num_exp(int(d), 5, P)
            test_num[1, i] = approx_beta_num_exp(int(d), 10, P)
            test_num[2, i] = approx_beta_num_exp(int(d), 20, P)
        store_norms[index, 0] = np.mean(test_num[0])
        store_norms[index, 1] = np.mean(test_num[1])
        store_norms[index, 2] = np.mean(test_num[2])
        index += 1
    return store_norms


def create_plot_avg_norm_grads(P, start_d, end_d, labels):
    """
    Plot of the average norm of the gradient for different d.
    """
    plt.clf()
    np.random.seed(start_d)
    store_norms = diff_d_norm_num_exp(P, start_d, end_d)
    plt.plot(np.arange(start_d, end_d), store_norms[:, 0], color='green')
    plt.plot(np.arange(start_d, end_d), store_norms[:, 1], color='purple')
    plt.plot(np.arange(start_d, end_d), store_norms[:, 2], color='blue')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.plot([], c='blue', label=labels[2])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 15})
    plt.xlabel(r"$d$", size=20)
    plt.xlim(start_d, end_d)
    plt.xticks(np.linspace(start_d, end_d, 4, endpoint=True))
    plt.tight_layout()
    plt.savefig('quad_P=%s_avg_norm_grads.png' % (P))


def create_plot_const_approx_beta(start_d, end_d):
    """
    Plot of const_approx_beta for different d.
    """
    plt.clf()
    c = 2.25
    plt.plot(np.arange(start_d, end_d),
             const_approx_beta(np.arange(start_d, end_d), c),
             color='blue')
    plt.xlabel(r"$d$", size=20)
    plt.xlim(start_d, end_d)
    plt.xticks(np.linspace(start_d, end_d, 4, endpoint=True))
    plt.tight_layout()
    plt.savefig('quad_const_approx_beta.png')


def create_plot_approx_beta(P, labels, start_d, end_d):
    """
    Plot of approx_beta for different d.
    """
    plt.clf()
    store_norms = diff_d_approx_beta_num_exp(P, start_d, end_d)
    plt.ylim(0, 0.5)
    plt.plot(np.arange(start_d, end_d), store_norms[:, 0], color='green')
    plt.plot(np.arange(start_d, end_d), store_norms[:, 1], color='purple')
    plt.plot(np.arange(start_d, end_d), store_norms[:, 2], color='blue')
    plt.plot([], c='green', label=labels[0])
    plt.plot([], c='purple', label=labels[1])
    plt.plot([], c='blue', label=labels[2])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 15})
    plt.xlabel(r"$d$", size=20)
    plt.xlim(start_d, end_d)
    plt.xticks(np.linspace(start_d, end_d, 4, endpoint=True))
    plt.tight_layout()
    plt.savefig('quad_approx_beta_diff_d.png')


if __name__ == "__main__":
    P = int(sys.argv[1])
    start_d = 5
    end_d = 200
    labels = [r'$\lambda_{max} = 5$', r'$\lambda_{max} = 10$',
              r'$\lambda_{max} = 20$']
    create_plot_avg_norm_grads(P, start_d, end_d, labels)
    create_plot_const_approx_beta(start_d, end_d)
    create_plot_approx_beta(P, labels, start_d, end_d)
