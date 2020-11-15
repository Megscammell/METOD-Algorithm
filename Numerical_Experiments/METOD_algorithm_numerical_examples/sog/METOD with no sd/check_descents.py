import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


def compute_boxplots(descents_50, met_list, d, p):
    """
    Create boxplots showing time taken by each method.

    Parameters
    ----------
    descents : Array of shape 100 x len(met_list)
                  Array containing number of extra descents for each set of
                  function parameters (row) for each method (column).
    met_list : list
               Contains the name of each method tested.
    p : integer
        Number of local minima.

    Returns
    -------
    Boxplot of number of extra descents taken by each method, saved as pdf.

    """    
    data_1 = descents_50[0]
    data_2 = descents_50[1]
    data_3 = descents_50[2]
    data_4 = descents_50[3]
    ticks = [r'$M=2$,''\n'r'$\beta=0.01$',r'$M=2$,''\n'r'$\beta=0.1$',
            r'$M=3$,''\n'r'$\beta=0.01$',r'$M=3$,''\n'r'$\beta=0.1$',
            r'$M=4$,''\n'r'$\beta=0.01$',r'$M=4$,''\n'r'$\beta=0.1$']
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(10, 5))
    bpl = plt.boxplot(data_1.T,
                      positions=np.array(range(len(data_1)))*4-0.9)
    bpcl = plt.boxplot(data_2.T,
                      positions=np.array(range(len(data_2)))*4.0-0.3)
    bpcr = plt.boxplot(data_3.T,
                      positions=np.array(range(len(data_3)))*4.0+0.3)
    bpr = plt.boxplot(data_4.T,
                      positions=np.array(range(len(data_4)))*4.0+0.9)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpcl, '#2C7BB6')
    set_box_color(bpcr, 'green')
    set_box_color(bpr, 'purple')
    plt.plot([], c='#D7191C', label=met_list[0])
    plt.plot([], c='#2C7BB6', label=met_list[1])
    plt.plot([], c='green', label=met_list[2])
    plt.plot([], c='purple', label=met_list[3])

    plt.legend(bbox_to_anchor=(0.99, 1.0175), loc='upper left',
               prop={'size': 10})
    plt.xticks(range(0, len(ticks)*4, 4), ticks, size=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Number of additional descents', size=10)
    plt.ylim(0,700)
    plt.tight_layout()
    plt.savefig('sog_descents_quad_d_%s_p_%s.pdf' % (d, p))


if __name__ == "__main__":
    """
    Checks that the number of local minima found by METOD in
    metod_numerical_exp_sog_no_sd.py and metod_numerical_exp_sog_with_sd.py
    are the same. Also checks that the number of local minima found by
    applying multistart is the same for all methods in met_list.
    """
    d = int(sys.argv[1])
    p = 50
    met_list = ['Nelder-Mead', 'Powell', 'Brent', 'Golden']
    beta_list = [0.01, 0.1]
    descents = np.zeros((len(met_list), 6, 100))
    index_met = 0
    for met in met_list:
        index_m = 0
        for j in range(2, 5):
            for beta in beta_list:
                df2 = pd.read_csv('sog_testing_minimize_met_%s_beta_%s'
                                    '_m=%s_d=%s_p=%s.csv' % (met, beta, j, d, p))
                descents[index_met, index_m, :] = df2['number_extra_descents_per_func_metod']
                index_m += 1
        index_met += 1
    compute_boxplots(descents, met_list, d, p)
