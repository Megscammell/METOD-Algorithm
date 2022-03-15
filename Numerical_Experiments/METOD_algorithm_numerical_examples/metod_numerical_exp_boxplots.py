import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def data_boxplots(func_name, d, p, num_p, beta_list, m_list, set_x, option,
                  initial_guess):
    """
    Store all results from numerical experiments.

    Parameters
    ----------
    func_name : string
                Name of function used to generate results.
    d : integer
        Size of dimension.
    p : integer
        Number of local minima.
    num_p : integer
            Number of starting points.
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
    set_x : string
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [2], which are randomly shuffled and used
            as starting points for the METOD algorithm.
    option : string
             Used to find the step size for each iteration of steepest
             descent.
             Choose from 'minimize' or 'minimize_scalar'. For more
             information on 'minimize' or 'minimize_scalar' see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.

    Returns
    ----------
    total_no_local_minimizers_metod : 3D array
                                      Contains the total number of local
                                      minimizers found by applying the METOD
                                      algorithm with various beta and warm up
                                      period m.
    total_no_local_minimizers_mult : 2D array
                                     Contains the total number of local
                                     minimizers found by applying the
                                     multistart.
    time_taken_metod : 3D array
                       Time taken by the METOD
                       algorithm for various beta and warm up
                       period m.
    time_taken_mult : 2D array
                      Time taken by the multistart.
    func_val_metod : 3D array
                     Contains the smallest function value observed by the
                     METOD algorithm with various beta and warm up
                     period m.
    func_val_mult : 2D array
                    Contains the smallest function value observed by the
                    multistart.
    extra_descents_metod : 3D array
                            Contains the total number of repeated local
                            descents from applying the METOD algorithm with
                            various beta and warm up period m.
    avg_grad : 3D array
               Norm of the gradient at each starting point of each function.
    grad_evals_metod : 3D array
                       Number of gradient evaluations computed by METOD with
                       various beta and warm up period m.
    grad_evals_mult : 2D array
                      Number of gradient evaluations computed by multistart.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    avg_grad = np.zeros((len(m_list) * len(beta_list), 100, num_p))
    grad_evals_metod = np.zeros((len(m_list) * len(beta_list), 100, num_p))
    total_no_local_minimizers_metod = np.zeros((len(m_list), len(beta_list),
                                               100))
    time_taken_metod = np.zeros((len(m_list), len(beta_list), 100))
    extra_descents_metod = np.zeros((len(m_list), len(beta_list), 100))
    func_val_metod = np.zeros((len(m_list), len(beta_list), 100))

    if p is not None and sigma_sq is not None:
        df_mult = pd.read_csv('%s_sd_metod_beta_0.1_m=2_d=%s_p=%s_%s_sig'
                              '_%s_%s_%s_%s.csv' %
                              (func_name, d, p, set_x, sigma_sq, num_p,
                               option, initial_guess))
        grad_evals_mult = np.genfromtxt('%s_grad_evals_mult_beta_0.1_m=2'
                                        '_d=%sp=%s_%s_sig'
                                        '_%s_%s_%s_%s.csv' %
                                        (func_name, d, p, set_x, sigma_sq,
                                         num_p, option, initial_guess),
                                        delimiter=',')

    elif p is not None:
        df_mult = pd.read_csv('%s_sd_metod_beta_0.1_m=2_d=%s_p=%s'
                              '_%s_%s_%s_%s.csv' %
                              (func_name, d, p, set_x, num_p,
                               option, initial_guess))
        grad_evals_mult = np.genfromtxt('%s_grad_evals_mult_beta_0.1_m=2_d=%s'
                                        '_p=%s_%s_%s_%s_%s.csv' %
                                        (func_name, d, p, set_x, num_p,
                                         option, initial_guess),
                                        delimiter=',')

    else:
        df_mult = pd.read_csv('%s_sd_metod_beta_0.01_m=2_d=%s'
                              '_%s_%s_%s_%s.csv' %
                              (func_name, d, set_x, num_p,
                               option, initial_guess))
        grad_evals_mult = np.genfromtxt('%s_grad_evals_mult_beta_0.01_m=2_d=%s'
                                        '_%s_%s_%s_%s.csv' %
                                        (func_name, d, set_x, num_p,
                                         option, initial_guess),
                                        delimiter=',')

    total_no_local_minimizers_mult = np.array(df_mult['number_minimizers_per_func_multistart'])
    time_taken_mult = np.array(df_mult['time_multistart'])
    func_val_mult = np.array(df_mult['min_func_val_multistart'])
    test = np.array(df_mult['number_minimizers_per_func_metod'])

    index_all = 0
    index_m = 0
    for m in m_list:
        index_beta = 0
        for beta in beta_list:
            if p is not None and sigma_sq is not None:
                df_metod = pd.read_csv('%s_metod_beta_%s_m=%s_d=%s_p=%s_%s_sig'
                                       '_%s_%s_%s_%s.csv' %
                                       (func_name, beta, m, d, p, set_x,
                                        sigma_sq, num_p, option,
                                        initial_guess))
                total_no_local_minimizers_metod[index_m, index_beta] = np.array(df_metod['number_minimizers_per_func_metod'])
                if beta == 0.1 and m == 2:
                    assert(np.all(total_no_local_minimizers_metod[index_m, index_beta] == test))

            elif p is not None:
                df_metod = pd.read_csv('%s_metod_beta_%s_m=%s_d=%s_p=%s_%s'
                                       '_%s_%s_%s.csv' %
                                       (func_name, beta, m, d, p, set_x, num_p,
                                        option, initial_guess))
                total_no_local_minimizers_metod[index_m, index_beta] = np.array(df_metod['number_minimizers_per_func_metod'])
                if beta == 0.1 and m == 2:
                    assert(np.all(total_no_local_minimizers_metod[index_m, index_beta] == test))
            else:
                df_metod = pd.read_csv('%s_metod_beta_%s_m=%s_d=%s_%s'
                                       '_%s_%s_%s.csv' %
                                       (func_name, beta, m, d, set_x, num_p,
                                        option, initial_guess))
                total_no_local_minimizers_metod[index_m, index_beta] = np.array(df_metod['number_minimizers_per_func_metod'])
                if beta == 0.01 and m == 2:
                    assert(np.all(total_no_local_minimizers_metod[index_m,
                                  index_beta] == test))
            time_taken_metod[index_m, index_beta] = np.array(df_metod['time_metod'])
            func_val_metod[index_m, index_beta] = np.array(df_metod['min_func_val_metod'])

            extra_descents_metod[index_m, index_beta] = np.array(df_metod['number_extra_descents_per_func_metod'])

            if p is not None and sigma_sq is not None:
                avg_grad[index_all] = np.genfromtxt('%s_grad_norm_beta_%s_m=%s'
                                                    '_d=%s_p=%s_%s_sig_%s_%s'
                                                    '_%s_%s.csv' %
                                                    (func_name, beta, m, d, p,
                                                     set_x, sigma_sq, num_p,
                                                     option, initial_guess),
                                                    delimiter=',')
                grad_evals_metod[index_all] = np.genfromtxt('%s_grad_evals_'
                                                            'metod_beta_%s_m=%s'
                                                            '_d=%s_p=%s_%s_sig'
                                                            '_%s_%s_%s_%s.csv' %
                                                            (func_name, beta,
                                                             m, d, p, set_x,
                                                             sigma_sq, num_p,
                                                             option,
                                                             initial_guess),
                                                            delimiter=',')

            elif p is not None:
                avg_grad[index_all] = np.genfromtxt('%s_grad_norm_beta_%s_m=%s'
                                                    '_d=%s_p=%s_%s_%s'
                                                    '_%s_%s.csv' %
                                                    (func_name, beta, m, d, p,
                                                     set_x, num_p, option,
                                                     initial_guess),
                                                    delimiter=',')
                grad_evals_metod[index_all] = np.genfromtxt('%s_grad_evals_'
                                                            'metod_beta_%s_m='
                                                            '%s_d=%s_p=%s_%s_'
                                                            '%s_%s_%s.csv' %
                                                            (func_name, beta,
                                                             m, d, p, set_x,
                                                             num_p, option,
                                                             initial_guess),
                                                            delimiter=',')

            else:
                avg_grad[index_all] = np.genfromtxt('%s_grad_norm_beta_%s_m=%s'
                                                    '_d=%s_%s_%s_%s_%s.csv' %
                                                    (func_name, beta, m, d,
                                                     set_x, num_p, option,
                                                     initial_guess),
                                                    delimiter=',')
                grad_evals_metod[index_all] = np.genfromtxt('%s_grad_evals_'
                                                            'metod_beta_%s_'
                                                            'm=%s_d=%s_%s_'
                                                            '%s_%s_%s.csv' %
                                                            (func_name, beta,
                                                             m, d, set_x,
                                                             num_p, option,
                                                             initial_guess),
                                                            delimiter=',')

            index_beta += 1
            index_all += 1
        index_m += 1

    check_avg_grad_and_sp_points(avg_grad, beta_list, m_list)

    return (total_no_local_minimizers_metod, total_no_local_minimizers_mult,
            time_taken_metod, time_taken_mult, func_val_metod, func_val_mult,
            extra_descents_metod, avg_grad, grad_evals_metod, grad_evals_mult)


def check_avg_grad_and_sp_points(avg_grad, beta_list, m_list):
    """
    Check norm of gradient at starting points for each function is the same
    for all combinations of beta and m.

    Parameters
    ----------
    avg_grad : 3D array
               Norm of the gradient at each starting point of each function.
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
    """
    for j in range(len(beta_list) * len(m_list)):
        for k in range(j+1, len(beta_list) * len(m_list)):
            assert(np.all(np.round(avg_grad[k], 5) ==
                   np.round(avg_grad[j], 5)))


def write_to_latex(arr, func_name, title, d, num_p, set_x,
                   option, initial_guess):
    """
    Write outputs to latex.

    Parameters
    ----------
    arr: array
         Convert arr to a Pandas dataframe and then write to latex.
    func_name : string
                Name of function used to generate results.
    title : string
            Name of saved outputs.
    d : integer
        Size of dimension.
    num_p : integer
            Number of starting points.
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    set_x : string
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [1], which are randomly shuffled and used
            as starting points for the METOD algorithm.
    option : string
             Used to find the step size for each iteration of steepest
             descent.
             Choose from 'minimize' or 'minimize_scalar'. For more
             information on 'minimize' or 'minimize_scalar' see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097
    """
    df = pd.DataFrame(arr)
    df.to_csv(df.to_csv('%s_%s_d=%s_%s_%s_%s_%s.csv' %
              (func_name, title, d, num_p, set_x,
               option, initial_guess)))
    with open('%s_%s_d=%s_%s_%s_%s_%s.tex' %
              (func_name, title, d, num_p, set_x,
               option, initial_guess), 'w') as tf:
        tf.write(df.to_latex())


def compute_tables(m_list, beta_list, total_no_local_minimizers_mult,
                   total_no_local_minimizers_metod, func_val_mult,
                   func_val_metod):
    """
    Compute summary tables of results and write to latex.

    Parameters
    ----------
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    total_no_local_minimizers_mult : 2D array
                                     Contains the total number of local
                                     minimizers found by applying the
                                     multistart.
    total_no_local_minimizers_metod : 3D array
                                      Contains the total number of local
                                      minimizers found by applying the METOD
                                      algorithm with various beta and warm up
                                      period m.
    func_val_mult : 2D array
                    Contains the smallest function value observed by the
                    multistart.
    func_val_metod : 3D array
                     Contains the smallest function value observed by the
                     METOD algorithm with various beta and warm up
                     period m.
    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    """
    same_global_min = np.zeros((len(m_list), len(beta_list)))
    minimizers_same = np.zeros((len(m_list), len(beta_list), 100))
    for i in range(len(m_list)):
        for j in range(len(beta_list)):
            assert(np.all(total_no_local_minimizers_mult >=
                   total_no_local_minimizers_metod[i, j]))
            same_global_min[i, j] = (np.where(np.round(func_val_mult, 2) ==
                                              np.round(func_val_metod[i, j],
                                              2))[0].shape[0]/100)
            minimizers_same[i, j] = (total_no_local_minimizers_mult -
                                     total_no_local_minimizers_metod[i, j])

    compute_freq_minimizers_found = np.zeros((len(m_list) * len(beta_list), 5))
    index = 0
    for i in range(len(m_list)):
        for j in range(len(beta_list)):
            compute_freq_minimizers_found[index, 0] = np.where(minimizers_same[i, j] == 0)[0].shape[0]
            compute_freq_minimizers_found[index, 1] = np.where(minimizers_same[i, j] == 1)[0].shape[0]
            compute_freq_minimizers_found[index, 2] = np.where(minimizers_same[i, j] == 2)[0].shape[0]
            compute_freq_minimizers_found[index, 3] = np.where(minimizers_same[i, j] >= 3)[0].shape[0]
            compute_freq_minimizers_found[index, 4] = same_global_min[i, j]
            index += 1

    write_to_latex(compute_freq_minimizers_found,
                   func_name, 'freq_minimizers',
                   d, num_p, set_x,  option, initial_guess)

    write_to_latex(np.round(same_global_min, 3), func_name,
                   'global_min', d, num_p, set_x,
                   option, initial_guess)


def function_info(func_name, d, p, set_x, sigma_sq, num_p, option, avg_grad):
    """
    Create summary tables for function information and write to latex.

    Parameters
    ----------
    func_name : string
                Name of function used to generate results.
    d : integer
        Size of dimension.
    p : integer
        Number of local minima.
    set_x : string
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [1], which are randomly shuffled and used
            as starting points for the METOD algorithm.
    sigma_sq : float
               Value of sigma squared.
    num_p : integer
            Number of starting points.
    option : string
             Used to find the step size for each iteration of steepest
             descent.
             Choose from 'minimize' or 'minimize_scalar'. For more
             information on 'minimize' or 'minimize_scalar' see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    avg_grad : 3D array
               Norm of the gradient at each starting point of each function.

    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    if p is not None and sigma_sq is not None:
        number_its = np.genfromtxt('%s_grad_evals_mult_beta_0.1_m=2_d=%sp=%s'
                                   '_%s_sig_%s_%s_%s_%s.csv' %
                                   (func_name, d, p, set_x, sigma_sq, num_p,
                                    option, initial_guess),
                                   delimiter=',')
    elif p is not None:
        number_its = np.genfromtxt('%s_grad_evals_mult_beta_0.1_m=2_d=%s_p=%s'
                                   '_%s_%s_%s_%s.csv' %
                                   (func_name, d, p, set_x, num_p, option,
                                    initial_guess),
                                   delimiter=',') - 1
    else:
        number_its = np.genfromtxt('%s_grad_evals_mult_beta_0.01_m=2_d=%s'
                                   '_%s_%s_%s_%s.csv' %
                                   (func_name, d, set_x, num_p, option,
                                    initial_guess),
                                   delimiter=',') - 1
    store_mean_its = np.zeros((100))
    store_min_its = np.zeros((100))
    store_max_its = np.zeros((100))
    store_mean_norm_grad = np.zeros((100))
    for j in range(100):
        store_mean_its[j] = np.mean(number_its[j])
        store_min_its[j] = np.min(number_its[j])
        store_max_its[j] = np.max(number_its[j])
        store_mean_norm_grad[j] = np.mean(avg_grad[0][j])
    results = np.array([np.round(np.mean(store_mean_its), 2),
                        np.round(np.min(store_min_its), 2),
                        np.round(np.max(store_max_its), 2),
                        np.round(np.mean(store_mean_norm_grad), 4)])
    write_to_latex(results, func_name, 'func_info', d, num_p, set_x,
                   option, initial_guess)


def set_box_color(bp, color):
    """Set colour for boxplot."""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def compute_prop_grad_evals(m_list, beta_list, grad_evals_metod,
                            grad_evals_mult):
    """
    Compute proportion of gradient evaluations computed by METOD
    compared to multistart.

    Parameters
    ----------
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    grad_evals_metod : 3D array
                       Number of gradient evaluations computed by METOD with
                       various beta and warm up period m.
    grad_evals_mult : 2D array
                      Number of gradient evaluations computed by multistart.

    Returns
    ----------
    prop_grad_evals : 2D array
                      Contains the proportion of gradient evaluations for
                      various beta and m.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    index_all = 0
    prop_grad_evals = np.zeros((len(m_list), len(beta_list), 100))
    for j in range(len(m_list)):
        for i in range(len(beta_list)):
            for func in range(100):
                prop_grad_evals[j, i, func] = (np.sum(grad_evals_metod[index_all, func]) /
                                               np.sum(grad_evals_mult[func]))
            index_all += 1
    return prop_grad_evals


def prop_missed_minimizers(beta_list, m_list,
                           total_no_local_minimizers_metod,
                           total_no_local_minimizers_mult,
                           func_name, d, num_p, set_x, option,
                           initial_guess):
    """
    Compute proportion of local minimizers missed by METOD
    compared to multistart and write outputs to latex.

    Parameters
    ----------
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
    total_no_local_minimizers_metod : 3D array
                                      Contains the total number of local
                                      minimizers found by applying the METOD
                                      algorithm with various beta and warm up
                                      period m.
    total_no_local_minimizers_mult : 2D array
                                     Contains the total number of local
                                     minimizers found by applying the
                                     multistart.
    func_name : string
                Name of function used to generate results.
    d : integer
        Size of dimension.
    num_p : integer
            Number of starting points.
    set_x : string
            If set_x = 'random', random starting points
            are generated for the METOD algorithm. If set_x = 'sobol'
            is selected, then a numpy.array with shape
            (num points * 2, d) of Sobol sequence samples are generated
            using SALib [2], which are randomly shuffled and used
            as starting points for the METOD algorithm.
    option : string
             Used to find the step size for each iteration of steepest
             descent.
             Choose from 'minimize' or 'minimize_scalar'. For more
             information on 'minimize' or 'minimize_scalar' see
             https://docs.scipy.org/doc/scipy/reference/optimize.html.
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    class_prop = np.zeros((len(m_list) * len(beta_list), 2))
    index = 0
    for i in range(len(m_list)):
        for j in range(len(beta_list)):
            class_prop[index, 0] = (np.round(
                                    (np.sum((
                                    total_no_local_minimizers_mult -
                                    total_no_local_minimizers_metod[i, j])) /
                                    np.sum(total_no_local_minimizers_mult)) * 100, 2))
            class_prop[index, 1] = (np.round(
                                    np.mean((total_no_local_minimizers_mult -
                                    total_no_local_minimizers_metod[i, j])),
                                    2))
            index += 1
    write_to_latex(class_prop, func_name, 'class_prop_info', d, num_p, set_x,
                   option, initial_guess)


def create_boxplots_ratio_m_1(arr1, labels, ticks, func_name, title, d, num_p,
                              set_x, p, sigma_sq, option, initial_guess):
    """
    Create boxplots
    """
    plt.figure(figsize=(5, 4))
    max_num = np.max(arr1)
    assert(max_num < 1.05)
    plt.ylim(0, 1.05)
    bpl = plt.boxplot(arr1.T)
    set_box_color(bpl, 'green')
    plt.plot([], c='green', label=labels[0])
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 15})
    plt.xlabel(r'$\beta$', size=16)
    plt.xticks(np.arange(1, len(ticks)+1), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if p is not None and sigma_sq is not None:
        plt.savefig('%s_%s_d=%s_%s_%s_p=%s_sig=%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     p, sigma_sq, option, initial_guess))
    elif p is not None:
        plt.savefig('%s_%s_d=%s_%s_%s_p=%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     p, option, initial_guess))
    else:
        plt.savefig('%s_%s_d=%s_%s_%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     option, initial_guess))


def create_boxplots_ratio_m_2(arr1, arr2, labels, ticks, func_name, title, d,
                              num_p, set_x, p, sigma_sq, option,
                              initial_guess):
    """
    Create boxplots
    """
    plt.figure(figsize=(5, 4))

    max_num = max(np.max(arr1), np.max(arr2))
    assert(max_num < 1.05)
    plt.ylim(0, 1.05)
    bpl = plt.boxplot(arr1.T,
                      positions=np.array(range(len(arr1)))*2.0-0.4)
    bpr = plt.boxplot(arr2.T,
                      positions=np.array(range(len(arr2)))*2.0+0.4)
    if labels == [r'$M =$ 1', r'$M =$ 2']:
        set_box_color(bpl, 'green')
        set_box_color(bpr, 'purple')
        plt.plot([], c='green', label=labels[0])
        plt.plot([], c='purple', label=labels[1])
        plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
                   prop={'size': 15})
        plt.xlabel(r'$\beta$', size=16)
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        if p is not None and sigma_sq is not None:
            plt.savefig('%s_%s_d=%s_%s_%s_p=%s_sig=%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         p, sigma_sq, option, initial_guess))
        elif p is not None:
            plt.savefig('%s_%s_d=%s_%s_%s_p=%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         p, option, initial_guess))
        else:
            plt.savefig('%s_%s_d=%s_%s_%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         option, initial_guess))
    elif labels == [r'$M =$ 2', r'$M =$ 3']:
        set_box_color(bpl, 'purple')
        set_box_color(bpr, 'navy')
        plt.plot([], c='purple', label=labels[0])
        plt.plot([], c='navy', label=labels[1])
        plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
                   prop={'size': 15})
        plt.xlabel(r'$\beta$', size=16)
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, size=15)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        if p is not None and sigma_sq is not None:
            plt.savefig('%s_%s_d=%s_%s_%s_p=%s_sig=%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         p, sigma_sq, option, initial_guess))
        elif p is not None:
            plt.savefig('%s_%s_d=%s_%s_%s_p=%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         p, option, initial_guess))
        else:
            plt.savefig('%s_%s_d=%s_%s_%s_%s_%s.png' %
                        (func_name, title, d, num_p, set_x,
                         option, initial_guess))


def create_boxplots_ratio_m_3(arr1, arr2, arr3, labels, ticks, func_name,
                              title, d, num_p, set_x, p, sigma_sq, option,
                              initial_guess):
    """
    Create boxplots
    """
    plt.figure(figsize=(5, 4))

    max_num = max(np.max(arr1), np.max(arr2), np.max(arr3))
    assert(max_num < 1.05)
    plt.ylim(0, 1.05)
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
    plt.legend(bbox_to_anchor=(0.99, 1.025), loc='upper left',
               prop={'size': 15})
    plt.xlabel(r'$\beta$', size=16)
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, size=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if p is not None and sigma_sq is not None:
        plt.savefig('%s_%s_d=%s_%s_%s_p=%s_sig=%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     p, sigma_sq, option, initial_guess))
    elif p is not None:
        plt.savefig('%s_%s_d=%s_%s_%s_p=%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     p, option, initial_guess))
    else:
        plt.savefig('%s_%s_d=%s_%s_%s_%s_%s.png' %
                    (func_name, title, d, num_p, set_x,
                     option, initial_guess))


if __name__ == "__main__":
    func_name = str(sys.argv[1])
    if func_name == 'sog':
        d = 20
        num_p = 100
        beta_list = [0.01, 0.1, 0.2]
        m_list = [2, 3]
        set_x = 'random'
        p = 10
        sigma_sq = 0.7
        option = 'm'
        initial_guess = 0.005
    elif func_name == 'shekel':
        d = 4
        num_p = 100
        beta_list = [0.01, 0.1, 0.2]
        m_list = [2, 3]
        set_x = 'random'
        p = 10
        sigma_sq = None
        option = 'm'
        initial_guess = 0.005
    elif func_name == 'quad':
        d = 100
        num_p = 1000
        beta_list = [0.01, 0.1, 0.2]
        m_list = [1, 2, 3]
        set_x = 'random'
        p = 50
        sigma_sq = None
        option = 'm'
        initial_guess = 0.005
    elif func_name == 'stybl_tang':
        d = 5
        num_p = 1000
        beta_list = [0.001, 0.005, 0.01]
        m_list = [1, 2]
        set_x = 'random'
        p = None
        sigma_sq = None
        option = 'm'
        initial_guess = 0.005
    elif func_name == 'qing':
        d = 5
        num_p = 1000
        beta_list = [0.005, 0.01, 0.05]
        m_list = [1, 2, 3]
        set_x = 'random'
        p = None
        sigma_sq = None
        option = 'm'
        initial_guess = 0.005
    elif func_name == 'hart':
        d = 6
        num_p = 50
        beta_list = [0.001, 0.01]
        m_list = [2, 3]
        set_x = 'random'
        p = None
        sigma_sq = None
        option = 'm'
        initial_guess = 0.005

    (total_no_local_minimizers_metod,
     total_no_local_minimizers_mult,
     time_taken_metod,
     time_taken_mult,
     func_val_metod,
     func_val_mult,
     extra_descents_metod,
     avg_grad,
     grad_evals_metod,
     grad_evals_mult) = data_boxplots(func_name, d, p, num_p, beta_list,
                                      m_list,  set_x, option, initial_guess)

    extra_descents_prop = ((extra_descents_metod +
                            total_no_local_minimizers_metod) /
                           (num_p))

    grad_evals_prop = compute_prop_grad_evals(m_list, beta_list,
                                              grad_evals_metod,
                                              grad_evals_mult)

    function_info(func_name, d, p, set_x, sigma_sq, num_p, option, avg_grad)

    compute_tables(m_list, beta_list, total_no_local_minimizers_mult,
                   total_no_local_minimizers_metod, func_val_mult,
                   func_val_metod)

    prop_missed_minimizers(beta_list, m_list,
                           total_no_local_minimizers_metod,
                           total_no_local_minimizers_mult,
                           func_name, d, num_p, set_x, option,
                           initial_guess)
    ticks = []
    labels = []
    for beta in beta_list:
        ticks.append(beta)
    for m in m_list:
        labels.append(r'$M =$ %s' % (m))

    if len(m_list) == 1:
        create_boxplots_ratio_m_1(extra_descents_prop[0], labels, ticks,
                                  func_name, 'ex_des_prop', d, num_p, set_x,
                                  p, sigma_sq, option, initial_guess)
        create_boxplots_ratio_m_1(grad_evals_prop[0], labels, ticks,
                                  func_name, 'grad_evals_prop', d, num_p,
                                  set_x, p, sigma_sq, option, initial_guess)
    elif len(m_list) == 2:
        create_boxplots_ratio_m_2(extra_descents_prop[0],
                                  extra_descents_prop[1], labels, ticks,
                                  func_name, 'ex_des_prop', d, num_p, set_x,
                                  p, sigma_sq, option, initial_guess)
        create_boxplots_ratio_m_2(grad_evals_prop[0],
                                  grad_evals_prop[1], labels, ticks,
                                  func_name, 'grad_evals_prop', d, num_p,
                                  set_x, p, sigma_sq, option, initial_guess)
    else:
        create_boxplots_ratio_m_3(extra_descents_prop[0],
                                  extra_descents_prop[1],
                                  extra_descents_prop[2],
                                  labels, ticks,
                                  func_name, 'ex_des_prop', d, num_p, set_x,
                                  p, sigma_sq, option, initial_guess)
        create_boxplots_ratio_m_3(grad_evals_prop[0],
                                  grad_evals_prop[1],
                                  grad_evals_prop[2],
                                  labels, ticks,
                                  func_name, 'grad_evals_prop', d, num_p,
                                  set_x, p, sigma_sq, option, initial_guess)
