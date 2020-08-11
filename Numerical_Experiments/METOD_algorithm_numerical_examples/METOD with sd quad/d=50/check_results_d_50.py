import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def box_plots_time(times, met_list, d, p):
    """
    Create boxplots showing time taken by each method.

    Parameters
    ----------
    times : Array of shape 100 x len(met_list)
            Array containing times for each set of function parameters (row)
            for each method (column)
    met_list : list
               Contains the name of each method tested
    d : integer
        Size of dimension.
    p : integer
        Number of local minima.

    Returns
    -------
    Boxplot of times taken by each method, saved as pdf.

    """

    plt.figure(figsize=(15, 5))
    plt.boxplot(times)
    xlabels = met_list
    plt.xticks(np.arange(1, len(met_list) + 1), xlabels, size=12)
    plt.yticks(size=12)
    plt.ylabel(r'Time (s)', size=12)

    plt.savefig('time_quad_d_%s_p_%s.pdf' %
                (d, p), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    """
    Checks that the number of local minima found by METOD in
    metod_numerical_exp_quad_no_sd.py and metod_numerical_exp_quad_with_sd.py
    are the same. Also checks that the number of local minimia found by
    applying multistart is the same for all methods in met_list.
    """

    d = 50
    beta = 0.1
    m = 1
    p = 50
    met_list = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'SLSQP', 'CG',
                'TNC', 'COBYLA', 'Brent', 'Golden']
    test_mets = np.zeros((len(met_list), 100))
    test_times = np.zeros((len(met_list), 100))
    index = 0
    for met in met_list:
        df2 = pd.read_csv('quad_sd_minimize_met_%s_beta_%s'
                          '_m=%s_d=%s_p=%s_All.csv' % (met, beta, m, d, p))
        df1 = pd.read_csv('quad_testing_minimize_met_%s_beta_%s'
                          '_m=%s_d=%s_p=%s_All.csv' % (met, beta, m, d, p))
        test_mets[index, :] = np.array(df2['number_minimas_'
                                           'per_func_multistart'])

        """
        Checks local minima found by metod is the same for
        metod_numerical_exp_quad_no_sd.py and
        metod_numerical_exp_quad_with_sd.py
        """
        assert(np.all(np.array(df2['number_minimas_per_func_metod']) ==
               np.array(df1['number_minimas_per_func_metod'])))
        test_times[index, :] = np.array(df2['time_multistart'])
        index += 1

    for j in range(len(met_list)):
        for i in range(j, len(met_list)):
            test_1 = test_mets[j]
            test_2 = test_mets[i]
            """
            Checks local minima found by multistart is the same for each method
            """
            assert(test_1.shape[0] == 100)
            assert(test_2.shape[0] == 100)
            assert(np.all(test_1 == test_2))

    box_plots_time(test_times.T, met_list, d, p)
