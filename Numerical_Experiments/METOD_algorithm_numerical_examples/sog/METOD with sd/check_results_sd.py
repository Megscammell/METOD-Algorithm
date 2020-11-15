import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


def compute_boxplots(times_50, times_100, met_list, p):
    """
    Create boxplots showing time taken by each method.

    Parameters
    ----------
    times_50 : Array of shape 100 x len(met_list)
               Array containing times for each set of function parameters (row)
               for each method (column) for dimension 50
    times_100 : Array of shape 100 x len(met_list)
               Array containing times for each set of function parameters (row)
               for each method (column) for dimension 100
    met_list : list
               Contains the name of each method tested
    p : integer
        Number of local minima.

    Returns
    -------
    Boxplot of times taken by each method, saved as pdf.

    """    
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(15, 5))
    bpl = plt.boxplot(times_50.T,
                      positions=np.array(range(len(times_50)))*3.0-0.4)
    bpr = plt.boxplot(times_100.T,
                      positions=np.array(range(len(times_100)))*3.0+0.4)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    plt.plot([], c='#D7191C', label=r'$d=50$')
    plt.plot([], c='#2C7BB6', label=r'$d=100$')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               prop={'size': 15})
    plt.xticks(range(0, len(met_list) * 3, 3), met_list, size=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('time_sog_d_p_%s.pdf' % (p))


if __name__ == "__main__":
    """
    Checks that the number of local minima found by METOD in
    metod_numerical_exp_sog_no_sd.py and metod_numerical_exp_sog_with_sd.py
    are the same. Also checks that the number of local minima found by
    applying multistart is the same for all methods in met_list.
    """
    beta = 0.01
    m = 1
    p = 50
    met_list = ['Nelder-Mead', 'Powell', 'Brent', 'Golden']
    test_mets_50 = np.zeros((len(met_list), 100))
    test_times_50 = np.zeros((len(met_list), 100))
    test_mets_100 = np.zeros((len(met_list), 100))
    test_times_100 = np.zeros((len(met_list), 100))
    index = 0
    for met in met_list:
        df2_50 = pd.read_csv('sog_testing_sd_met_%s_beta_%s'
                             '_m=%s_d=%s_p=%s.csv' % (met, beta, m, 50, p))
        df1_50 = pd.read_csv('sog_testing_minimize_met_%s_beta_%s'
                             '_m=%s_d=%s_p=%s.csv' % (met, beta, m, 50, p))
        test_mets_50[index, :] = np.array(df2_50['number_minimas_'
                                                 'per_func_multistart'])
        df2_100 = pd.read_csv('sog_testing_sd_met_%s_beta_%s'
                              '_m=%s_d=%s_p=%s.csv' % (met, beta, m, 100, p))
        df1_100 = pd.read_csv('sog_testing_minimize_met_%s_beta_%s'
                              '_m=%s_d=%s_p=%s.csv' % (met, beta, m, 100, p))
        test_mets_100[index, :] = np.array(df2_100['number_minimas_'
                                           'per_func_multistart'])
        """
        Checks local minima found by metod is the same for
        metod_numerical_exp_sog_no_sd.py and
        metod_numerical_exp_sog_with_sd.py
        """
        assert(np.all(np.array(df2_50['number_minimas_per_func_metod']) ==
               np.array(df1_50['number_minimas_per_func_metod'])))
        assert(np.all(np.array(df2_100['number_minimas_per_func_metod']) ==
               np.array(df1_100['number_minimas_per_func_metod'])))
        test_times_50[index, :] = np.array(df2_50['time_multistart'])
        test_times_100[index, :] = np.array(df2_100['time_multistart'])
        index += 1

    for j in range(len(met_list)):
        for i in range(j, len(met_list)):
            test_1_50 = test_mets_50[j]
            test_2_50 = test_mets_50[i]
            test_1_100 = test_mets_100[j]
            test_2_100 = test_mets_100[i]
            """
            Checks local minima found by multistart is the same for each method
            """
            assert(test_1_50.shape[0] == 100)
            assert(test_2_50.shape[0] == 100)
            assert(test_1_100.shape[0] == 100)
            assert(test_2_100.shape[0] == 100)
            assert(np.all(test_1_50 == test_2_50))
            assert(np.all(test_1_100 == test_2_100))

    compute_boxplots(test_times_50, test_times_100, met_list, p)

