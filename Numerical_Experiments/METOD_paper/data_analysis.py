import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def creating_tables(beta_list, m_list, func_name, d, p, set_x, sigma_sq,
                    num_p, option, initial_guess, type_func, type_results):
    """
    Store all results from numerical experiments.

    Parameters
    ----------
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
    m_list : list
             Contains various values for the warm up period in METOD [1] used
             to obtain results.
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
            using SALib [2], which are randomly shuffled and used
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
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    type_func : string
                Either type_func = 'new' to obtain results in thesis or
                type_func = 'old' to obtain results in [1].

    Returns
    ----------
    data_table_pd : pandas dataframe
                    Summary data from METOD.
    multistart_pd : pandas dataframe
                    Summary data from multistart.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    if type_func == 'new':
        data_table = np.zeros((len(m_list) * len(beta_list), 9))
    else:
        data_table = np.zeros((len(m_list) * len(beta_list), 10))
    multistart_table = np.zeros((3))
    index = 0
    for beta in beta_list:
        for m in m_list:

            if func_name == 'quad':
                if type_func == 'new':
                    if m == 1 and beta == 0.2:
                        df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                          '%s_%s_%s_%s_%s.csv'
                                          % (func_name, beta, m, d, p, set_x,
                                             num_p, option[0], initial_guess,
                                             type_func)))
                    else:
                        df = pd.read_csv(('%s_metod_beta_%s_m=%s_d=%s_p=%s_'
                                          '%s_%s_%s_%s_%s.csv'
                                          % (func_name, beta, m, d, p, set_x,
                                             num_p, option[0], initial_guess,
                                             type_func)))

                else:
                    df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                      '%s_%s_%s_%s_%s.csv'
                                      % (func_name, beta, m, d, p, set_x,
                                         num_p, option[0], initial_guess,
                                         type_func)))
                data_table[index, 0] = np.where(np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 1] = np.where(np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 2] = np.where(np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]

                data_table[index, 3] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 4] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 5] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]
                if type_func == 'old':
                    data_table[index, 6] = np.sum(np.array(df["prop_class"]))
                    data_table[index, 7] = np.sum(np.array(df["total_times_minimizer_missed"]))
                    data_table[index, 8] = np.sum(np.array(df["total_no_times_inequals_sat"]))
                    data_table[index, 9] = np.round((data_table[index, 7] / data_table[index, 8]) * 100, 3)
                else:
                    data_table[index, 6] = np.sum(np.array(df["total_times_minimizer_missed"]))
                    data_table[index, 7] = np.sum(np.array(df["total_no_times_inequals_sat"]))
                    data_table[index, 8] = np.round((data_table[index, 6] / data_table[index, 7]) * 100, 3)
                if type_func == 'old':
                    if type_results == 'thesis':
                        if beta == 0.01 and m == 1:
                            check_multistart = np.array(df['number_minimizers_per_func_multistart'])
                    else:
                        if beta == 0.005 and m == 2:
                            check_multistart = np.array(df['number_minimizers_per_func_multistart'])
                    assert(np.all(check_multistart ==
                                  np.array(df['number_minimizers_per_func_multistart'])))

                index += 1

            elif func_name == 'sog':
                if type_func == 'new':
                    if m == 2 and beta == 0.2 and d == 20:
                        df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                          '_%s_sig_%s_%s_%s_%s_%s.csv'
                                          % (func_name, beta, m, d, p, set_x,
                                             sigma_sq, num_p, option[0],
                                             initial_guess, type_func)))
                    elif m == 3 and beta == 0.2 and d == 50:
                        df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                          '_%s_sig_%s_%s_%s_%s_%s.csv'
                                          % (func_name, beta, m, d, p, set_x,
                                             sigma_sq, num_p, option[0],
                                             initial_guess, type_func)))
                    else:
                        df = pd.read_csv(('%s_metod_beta_%s_m=%s_d=%s_p=%s_%s'
                                          '_sig_%s_%s_%s_%s_%s.csv'
                                          % (func_name, beta, m, d, p, set_x,
                                             sigma_sq, num_p, option[0],
                                             initial_guess, type_func)))
                else:
                    df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s'
                                      '_sig_%s_%s_%s_%s_%s.csv'
                                      % (func_name, beta, m, d, p, set_x,
                                         sigma_sq, num_p, option[0],
                                         initial_guess, type_func)))

                data_table[index, 0] = np.where(np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 1] = np.where(np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 2] = np.where(np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]

                data_table[index, 3] = np.min(np.array(df["number_extra_descents_per_func_metod"]) +
                                              np.array(df['number_minimizers_per_func_metod']))
                data_table[index, 4] = np.max(np.array(df["number_extra_descents_per_func_metod"]) +
                                              np.array(df['number_minimizers_per_func_metod']))
                data_table[index, 5] = np.mean(np.array(df["number_extra_descents_per_func_metod"]) +
                                               np.array(df['number_minimizers_per_func_metod']))

                if type_func == 'old':
                    data_table[index, 6] = np.sum(np.array(df["prop_class"]))
                    data_table[index, 7] = np.sum(np.array(df["total_times_minimizer_missed"]))
                    data_table[index, 8] = np.sum(np.array(df["total_no_times_inequals_sat"]))
                    data_table[index, 9] = np.round((data_table[index, 7] / data_table[index, 8]) * 100, 3)
                else:
                    data_table[index, 6] = np.sum(np.array(df["total_times_minimizer_missed"]))
                    data_table[index, 7] = np.sum(np.array(df["total_no_times_inequals_sat"]))
                    data_table[index, 8] = np.round((data_table[index, 6] / data_table[index, 7]) * 100, 3)
                if type_func == 'old':
                    if type_results == 'thesis':
                        if beta == 0.01 and m == 1:
                            check_multistart = np.array(df['number_minimizers_per_func_multistart'])
                    else:
                        if beta == 0.005 and m == 2:
                            check_multistart = np.array(df['number_minimizers_per_func_multistart'])
                    assert(np.all(check_multistart ==
                                  np.array(df['number_minimizers_per_func_multistart'])))

                index += 1
    if type_func == 'new':
        if func_name == 'quad':
            df_mult = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                   '%s_%s_%s_%s_%s.csv'
                                   % (func_name, 0.2, 1, d, p, set_x,
                                      num_p, option[0], initial_guess,
                                      type_func)))
            check_multistart = np.array(df_mult['number_minimizers_per_func_multistart'])
        else:
            if d == 20:
                df_mult = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                       '_%s_sig_%s_%s_%s_%s_%s.csv'
                                      % (func_name, 0.2, 2, d, p, set_x,
                                         sigma_sq, num_p, option[0],
                                         initial_guess, type_func)))
            elif d == 50:
                df_mult = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                       '_%s_sig_%s_%s_%s_%s_%s.csv'
                                       % (func_name, 0.2, 3, d, p, set_x,
                                          sigma_sq, num_p, option[0],
                                          initial_guess, type_func)))
            check_multistart = np.array(df_mult['number_minimizers_per_func_multistart'])

    multistart_table[0] = np.where(check_multistart == p)[0].shape[0]
    multistart_table[1] = np.where(check_multistart == p - 1)[0].shape[0]
    multistart_table[2] = np.where(check_multistart <= p - 2)[0].shape[0]

    m_list_pd = [x for b in beta_list for x in m_list]

    if func_name == 'quad':
        if type_func == 'old':
            data_table_pd = pd.DataFrame(data=data_table,
                                         index=m_list_pd,
                                         columns=["Regions Missed = 0",
                                                  "Regions Missed = 1",
                                                  "Regions Missed >= 2",
                                                  "Descents = p",
                                                  "Descents = p-1",
                                                  "Descents <= p-2",
                                                  "Percentage of incorrectly identified regions",
                                                  "Total Minimizers missed",
                                                  "Total times sat inequality",
                                                  "Percentage minimizers missed"])
        else:
            data_table_pd = pd.DataFrame(data=data_table,
                                         index=m_list_pd,
                                         columns=["Regions Missed = 0",
                                                  "Regions Missed = 1",
                                                  "Regions Missed >= 2",
                                                  "Descents = p",
                                                  "Descents = p-1",
                                                  "Descents <= p-2",
                                                  "Total Minimizers missed",
                                                  "Total times sat inequality",
                                                  "Percentage minimizers missed"])
    else:
        if type_func == 'old':
            data_table_pd = pd.DataFrame(data=data_table,
                                         index=m_list_pd,
                                         columns=["Regions Missed = 0",
                                                  "Regions Missed = 1",
                                                  "Regions Missed >= 2",
                                                  "Min descents",
                                                  "Max descents",
                                                  "Avg descents",
                                                  "Percentage of incorrectly identified regions",
                                                  "Total Minimizers missed",
                                                  "Total times sat inequality",
                                                  "Percentage minimizers missed"])
        else:
            data_table_pd = pd.DataFrame(data=data_table,
                                         index=m_list_pd,
                                         columns=["Regions Missed = 0",
                                                  "Regions Missed = 1",
                                                  "Regions Missed >= 2",
                                                  "Min descents",
                                                  "Max descents",
                                                  "Avg descents",
                                                  "Total Minimizers missed",
                                                  "Total times sat inequality",
                                                  "Percentage minimizers missed"])

    multistart_pd = pd.DataFrame(data=multistart_table)
    return data_table_pd, multistart_pd


def write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq,
                   num_p, option, initial_guess, type_func, name):
    """
    Write outputs to latex.

    Parameters
    ----------
    data_table_pd : pandas dataframe
                    Summary data from METOD.
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
    initial_guess : float or integer (optional)
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    type_func : string
                Either type_func = 'new' to obtain results in thesis or
                type_func = 'old' to obtain results in [1].
    name: string
          Name of saved outputs.

    References
    ----------
    1) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097
    """
    if func_name == 'quad':
        data_table_pd.to_csv('%s_%s_d=%s_p=%s_%s_%s_%s_%s_%s.csv'
                             % (func_name, name, d, p, set_x, num_p,
                                option[0], initial_guess, type_func))
        with open('%s_%s_d=%s_p=%s_%s_%s_%s_%s_%s.tex'
                  % (func_name, name, d, p, set_x, num_p, option[0],
                     initial_guess, type_func), 'w') as tf:
            tf.write(data_table_pd.to_latex())

    elif func_name == 'sog':
        data_table_pd.to_csv('%s_%s_d=%s_p=%s_%s_%s_%s'
                             '_%s_%s_%s.csv'
                             % (func_name,  name, d, p, set_x, sigma_sq, num_p,
                                option[0], initial_guess, type_func))
        with open('%s_%s_d=%s_p=%s_%s_%s_%s_%s_%s_%s.tex'
                  % (func_name,  name, d, p, set_x, sigma_sq, num_p, option[0],
                     initial_guess, type_func), 'w') as tf:
            tf.write(data_table_pd.to_latex())


def frequency_of_descents(func_name, d, p, beta_list, m_list, set_x, sigma_sq,
                          num_p, option, initial_guess, type_func):
    """
    Compute and save the number of points that satisfied [1, Eq. 9] for more
    than one region of attraction for various beta and m.

    Parameters
    ----------
    func_name : string
                Name of function used to generate results.
    d : integer
        Size of dimension.
    p : integer
        Number of local minima.
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
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    type_func : string
                Either type_func = 'new' to obtain results in thesis or
                type_func = 'old' to obtain results in [1].


    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    freq_table = np.zeros((len(beta_list), len(m_list)))
    index_m = 0
    for m in m_list:
        index_beta = 0
        for beta in beta_list:
            if func_name == 'quad':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                  '%s_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x,
                                     num_p, option[0], initial_guess,
                                     type_func)))
            elif func_name == 'sog':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                  '_%s_sig_%s_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x, sigma_sq,
                                     num_p, option[0], initial_guess,
                                     type_func)))
            freq_table[index_beta, index_m] = np.sum(np.array(df['greater_than_one_region']))
            index_beta += 1
        index_m += 1
    np.savetxt('freq_table_d=%s'
               '_p=%s_%s_%s_%s_%s_%s.csv' %
               (d, p, set_x, num_p, option[0], initial_guess, type_func),
               freq_table,
               delimiter=',')
    return freq_table


def produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess,
                                    type_func, type_results):
    """
    Produce bar charts showing the number of points which satisfied [1, Eq. 9]
    for more than one region of attraction for various beta and m.

    Parameters
    ----------
    beta_list : list
                Contains various values of beta used in METOD [1] to obtain
                results.
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
            using SALib [2], which are randomly shuffled and used
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
    initial_guess : float or integer
                    Initial guess passed to scipy.optimize.minimize and the
                    upper bound for the bracket interval when using the
                    'Brent' or 'Golden' method for
                    scipy.optimize.minimize_scalar. This
                    is recommended to be small.
    type_func : string
                Either type_func = 'new' to obtain results in thesis or
                type_func = 'old' to obtain results in [1].

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Herman et al, (2017), SALib: An open-source Python library for
       Sensitivity Analysis, Journal of Open Source Software, 2(9), 97, doi:10.
       21105/joss.00097

    """
    plt.figure(figsize=(5, 5))
    x = np.arange(1, len(beta_list) + 1)
    w = 0.4
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.bar(x - w, freq_table[:, 0], width=w, color='purple', zorder=2)
    plt.bar(x, freq_table[:, 1], width=w, color='blue', zorder=2)
    plt.xticks(x + (w * - 0.52), beta_list, size=14)
    plt.yticks(fontsize=15)
    plt.xlabel(r'$\beta $', size=16)

    if type_results == 'thesis' and func_name == 'quad':
        purple_patch = mpatches.Patch(color='purple', label=r'$M=1$')
        blue_patch = mpatches.Patch(color='blue', label=r'$M=2$')
    else:
        purple_patch = mpatches.Patch(color='purple', label=r'$M=2$')
        blue_patch = mpatches.Patch(color='blue', label=r'$M=3$')

    plt.legend(handles=[purple_patch, blue_patch], loc='upper right',
               bbox_to_anchor=(1, 1), borderaxespad=0.,
               prop={'size': 15})
    plt.grid(axis='y')
    plt.savefig('%s_metod_freq_descents_graphs_d=%s_p=%s_%s_%s_%s_%s_%s.png'
                % (func_name, d, p, set_x, num_p, option[0], initial_guess,
                   type_func),
                bbox_inches='tight')


if __name__ == "__main__":
    func_name = str(sys.argv[1])
    type_func = str(sys.argv[2])
    d = int(sys.argv[3])
    type_results = str(sys.argv[4])
    if func_name == 'quad':
        p = 50
        set_x = 'random'
        num_p = 1000
        option = 'minimize'
        sigma_sq = None
        if type_func == 'old':
            if type_results == 'paper':
                initial_guess = 0.05
                m_list = [2, 3]
                beta_list = [0.005, 0.01, 0.05, 0.1]
            elif type_results == 'thesis':
                initial_guess = 0.005
                m_list = [1, 2]
                beta_list = [0.01, 0.1, 0.2]
        else:
            initial_guess = 0.005
            m_list = [1, 2, 3]
            beta_list = [0.2]
    elif func_name == 'sog':
        set_x = 'random'
        num_p = 1000
        option = 'minimize'
        m_list = [2, 3, 4]
        initial_guess = 0.005
        if type_func == 'old':
            p = 20
            beta_list = [0.005, 0.01, 0.05, 0.1]
            if d == 50:
                sigma_sq = 1.3
            elif d == 100:
                sigma_sq = 4
        else:
            p = 10
            beta_list = [0.2]
            if d == 20:
                sigma_sq = 0.7
            elif d == 50:
                sigma_sq = 1.6

    data_table_pd, multistart_pd = (creating_tables(
                                    beta_list, m_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess,
                                    type_func, type_results))
    write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq, num_p,
                   option, initial_guess, type_func, 'metod_table')

    write_to_latex(multistart_pd, func_name, d, p, set_x, sigma_sq, num_p,
                   option, initial_guess, type_func, 'mult_data')
    if type_func == 'old':
        freq_table = frequency_of_descents(func_name, d, p, beta_list, m_list,
                                           set_x, sigma_sq, num_p, option,
                                           initial_guess, type_func)
        produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x,
                                        sigma_sq, num_p, option, initial_guess,
                                        type_func, type_results)
