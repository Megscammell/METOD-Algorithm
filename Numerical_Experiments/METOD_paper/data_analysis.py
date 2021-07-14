import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
%matplotlib inline
import seaborn as sns
import sys
from matplotlib import rc


def creating_tables(beta_list, m_list, func_name, d, p, set_x, sigma_sq, num_p, option, initial_guess):
    
    data_table = np.zeros((len(m_list) * len(beta_list), 7))
    index = 0
    for beta in beta_list:
        for m in m_list:
            
            if func_name == 'quad':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x,
                                     num_p, option[0], initial_guess)))
            elif func_name == 'sog':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s_sig_%s_%s_%s_%ss.csv'
                                  % (func_name, beta, m, d, p, set_x, sigma_sq,
                                     num_p, option[0], initial_guess)))

            data_table[index, 0] = np.where(np.array(df['Number regions']) == p)[0].shape[0] 
            data_table[index, 1] = np.where(np.array(df['Number regions']) == p - 1)[0].shape[0]  
            data_table[index, 2] = np.where(np.array(df['Number regions']) <= p - 2)[0].shape[0] 
            
            data_table[index, 3] = np.where(np.array(df['Total Descents']) == p)[0].shape[0] 
            data_table[index, 4] = np.where(np.array(df['Total Descents']) == p- 1)[0].shape[0] 
            data_table[index, 5] = np.where(np.array(df['Total Descents']) <= p- 2)[0].shape[0] 
            # Only correct if 100 functions
            data_table[index, 6] = np.sum(np.array(df["Misclassifications"]) /1000)
            index += 1
    m_list_pd = [x for l in beta_list for x in m_list]
    data_table_pd = pd.DataFrame(data = data_table,
                                 index=m_list_pd,
                                 columns = ["Regions Missed = 0",
                                           "Regions Missed = 1",
                                           "Regions Missed >= 2",
                                            "Descents = p",
                                           "Descents = p-1",
                                           "Descents <= p-2",
                                           "Proportion of incorrectly identified regions"])
    return data_table_pd


def write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq, num_p, option, initial_guess):
    if func_name == 'quad':
        data_table_pd.to_csv('%s_data_table_d=%s_p=%s_%s_%s_%s_%s.csv'
                             % (func_name, p, set_x, num_p, option[0], initial_guess))
        with open('data_table_d=%s_p=%s_%s_%s_%s_%s.tex' 
                  % (d, num_p, p, set_x, option[0], initial_guess), 'w') as tf:
            tf.write(data_table_pd.to_latex())
            
    elif func_name == 'sog':
        data_table_pd.to_csv('%s_data_table_d=%s_p=%s_%s_%s_%s_%s_%s.csv'
                             % (func_name, p, set_x, sigma_sq, num_p, option[0], initial_guess))
        with open('%s_data_table_d=%s_p=%s_%s_%s_%s_%s_%s.tex' 
                  % (func_name, d, num_p, p, set_x, sigma_sq, option[0], initial_guess), 'w') as tf:
            tf.write(data_table_pd.to_latex())


def frequency_of_descents(func_name, d, p, beta_list, m_list, set_x, sigma_sq, num_p, option, initial_guess):
    freq_table = np.zeros((len(beta_list), len(m_list)))
    index_m = 0
    for m in m_list:
        index_beta = 0
        for beta in beta_list:
            if func_name == 'quad':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s_%s_%s_%s.csv'
                                  % (func_name, beta, m, d, p, set_x,
                                     num_p, option[0], initial_guess)))
            elif func_name == 'sog':
                df = pd.read_csv(('%s_sd_metod_beta_%s_m=%s_d=%s_p=%s_%s_sig_%s_%s_%s_%ss.csv'
                                  % (func_name, beta, m, d, p, set_x, sigma_sq,
                                     num_p, option[0], initial_guess)))        
            freq_table[index_beta, index_m] = np.sum(np.array(df['Number of points sat more than one region']))
            index_beta += 1
        index_m += 1
    np.savetxt('freq_table_d=%s'
                '_p=%s_%s_%s_%s_%s.csv' %
                (d, p, set_x, num_p, option[0], initial_guess),
                freq_table,
                delimiter=',')
    return freq_table


def produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x, sigma_sq, num_p, option, initial_guess):

    y1=freq_table[:, 0]     
    y2=freq_table[:, 1] 
    x = np.arange(1, len(beta_list) + 1)
    w = 0.4
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.bar(x - w, freq_table[:, 0], width=w, color='purple', zorder=2)
    plt.bar(x, freq_table[:, 1], width=w, color='blue', zorder=2)
    plt.xticks(x + (w * - 0.52), beta_list)
    plt.xlabel(r'$\beta $')

    #legend
    purple_patch = mpatches.Patch(color='purple', label=r'$M=2$')
    blue_patch = mpatches.Patch(color='blue', label=r'$M=3$')

    lgd = plt.legend(handles=[purple_patch, blue_patch], loc='upper right', bbox_to_anchor=(1, 1),borderaxespad=0.)
    plt.grid(axis='y')
    plt.savefig('%s_metod_freq_descents_graphs_d=%s_p=%s_%s_%s_%s_%s_%s.png'
                % (func_name, d, p, set_x, num_p, option[0], initial_guess),
                bbox_inches='tight')


if __name__ == "__main__":
    f = prev_mt_alg.quad_function
    g = prev_mt_alg.quad_gradient
    check_func = prev_mt_alg.calc_minimizer_quad

    func_name = str(sys.argv[1])
    d = int(sys.argv[2])
    p = int(sys.argv[3])
    set_x = str(sys.argv[4])
    sigma_sq = float(sys.argv[5])
    num_p = int(sys.argv[6])
    option = str(sys.argv[7])
    initial_guess = float(sys.argv[8])

    beta_list = [0.005,0.01, 0.05, 0.1]
    m_list = [2,3]

    data_table_pd = creating_tables(beta_list, m_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess)
    write_to_latex(data_table_pd, func_name, d, p, set_x, sigma_sq, num_p,
                   option, initial_guess)
    freq_table = frequency_of_descents(func_name, d, p, beta_list, m_list,
                                       set_x, sigma_sq, num_p, option,
                                       initial_guess)
    produce_freq_of_descents_graphs(beta_list, func_name, d, p, set_x,
                                    sigma_sq, num_p, option, initial_guess)