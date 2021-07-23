import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def creating_tables(beta_list, m_list, func_name, d, p, set_x, sigma_sq,
                    num_p, option, initial_guess):

    data_table = np.zeros((len(m_list) * len(beta_list), 7))
    index = 0
    for beta in beta_list:
        for m in m_list:

            if func_name == 'quad':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                  '%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x,
                                     num_p, option[0], initial_guess)))
                data_table[index, 0] = np.where(np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 1] = np.where(np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 2] = np.where(np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]

                data_table[index, 3] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 4] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 5] = np.where(np.array(df["number_extra_descents_per_func_metod"]) +
                                                np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]

                data_table[index, 6] = np.sum(np.array(df["prop_class"]))
                index += 1

            elif func_name == 'sog':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s_sig_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x, sigma_sq,
                                     num_p, option[0], initial_guess)))

                data_table[index, 0] = np.where(np.array(df['number_minimizers_per_func_metod']) == p)[0].shape[0]
                data_table[index, 1] = np.where(np.array(df['number_minimizers_per_func_metod']) == p - 1)[0].shape[0]
                data_table[index, 2] = np.where(np.array(df['number_minimizers_per_func_metod']) <= p - 2)[0].shape[0]

                data_table[index, 3] = np.min(np.array(df["number_extra_descents_per_func_metod"]) +
                                              np.array(df['number_minimizers_per_func_metod']))
                data_table[index, 4] = np.max(np.array(df["number_extra_descents_per_func_metod"]) +
                                              np.array(df['number_minimizers_per_func_metod']))
                data_table[index, 5] = np.mean(np.array(df["number_extra_descents_per_func_metod"]) +
                                               np.array(df['number_minimizers_per_func_metod']))

                data_table[index, 6] = np.sum(np.array(df["prop_class"]))
                index += 1
    m_list_pd = [x for b in beta_list for x in m_list]
    data_table_pd = pd.DataFrame(data=data_table,
                                 index=m_list_pd,
                                 columns=["Regions Missed = 0",
                                          "Regions Missed = 1",
                                          "Regions Missed >= 2",
                                          "Descents = p",
                                          "Descents = p-1",
                                          "Descents <= p-2",
                                          "Proportion of incorrectly identified regions"])
    return data_table_pd


def write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq,
                   num_p, option, initial_guess):
    if func_name == 'quad':
        data_table_pd.to_csv('%s_data_table_d=%s_p=%s_%s_%s_%s_%s.csv'
                             % (func_name, d, p, set_x, num_p,
                                option[0], initial_guess))
        with open('%s_data_table_d=%s_p=%s_%s_%s_%s_%s.tex'
                  % (func_name, d, p, set_x, num_p, option[0],
                     initial_guess), 'w') as tf:
            tf.write(data_table_pd.to_latex())

    elif func_name == 'sog':
        data_table_pd.to_csv('%s_data_table_d=%s_p=%s_%s_%s_%s_%s_%s.csv'
                             % (func_name, d, p, set_x, sigma_sq, num_p,
                                option[0], initial_guess))
        with open('%s_data_table_d=%s_p=%s_%s_%s_%s_%s_%s.tex'
                  % (func_name, d, p, set_x, sigma_sq, num_p, option[0],
                     initial_guess), 'w') as tf:
            tf.write(data_table_pd.to_latex())


def frequency_of_descents(func_name, d, p, beta_list, m_list, set_x, sigma_sq,
                          num_p, option, initial_guess):
    freq_table = np.zeros((len(beta_list), len(m_list)))
    index_m = 0
    for m in m_list:
        index_beta = 0
        for beta in beta_list:
            if func_name == 'quad':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_'
                                  '%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x,
                                     num_p, option[0], initial_guess)))
            elif func_name == 'sog':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s'
                                  '_%s_sig_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x, sigma_sq,
                                     num_p, option[0], initial_guess)))
            freq_table[index_beta, index_m] = np.sum(np.array(df['greater_than_one_region']))
            index_beta += 1
        index_m += 1
    np.savetxt('freq_table_d=%s'
               '_p=%s_%s_%s_%s_%s.csv' %
               (d, p, set_x, num_p, option[0], initial_guess),
               freq_table,
               delimiter=',')
    return freq_table


def produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess):

    x = np.arange(1, len(beta_list) + 1)
    w = 0.4
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.bar(x - w, freq_table[:, 0], width=w, color='purple', zorder=2)
    plt.bar(x, freq_table[:, 1], width=w, color='blue', zorder=2)
    plt.xticks(x + (w * - 0.52), beta_list)
    plt.xlabel(r'$\beta $')

    purple_patch = mpatches.Patch(color='purple', label=r'$M=2$')
    blue_patch = mpatches.Patch(color='blue', label=r'$M=3$')

    lgd = plt.legend(handles=[purple_patch, blue_patch], loc='upper right',
                     bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.grid(axis='y')
    plt.savefig('%s_metod_freq_descents_graphs_d=%s_p=%s_%s_%s_%s_%s.png'
                % (func_name, d, p, set_x, num_p, option[0], initial_guess),
                bbox_inches='tight')


if __name__ == "__main__":
    func_name = 'quad'
    d = 100
    p = 50
    set_x = 'random'
    num_p = 1000
    option = 'minimize'
    initial_guess = 0.05
    sigma_sq = None

    beta_list = [0.005, 0.01, 0.05, 0.1]
    m_list = [2, 3]

    data_table_pd = creating_tables(beta_list, m_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess)
    write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq, num_p,
                   option, initial_guess)
    freq_table = frequency_of_descents(func_name, d, p, beta_list, m_list,
                                       set_x, sigma_sq, num_p, option,
                                       initial_guess)
    produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess)
