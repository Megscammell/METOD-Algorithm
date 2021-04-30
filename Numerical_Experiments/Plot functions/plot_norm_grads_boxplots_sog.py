import numpy as np
import sys
import matplotlib.pyplot as plt; plt.rcdefaults()
from numpy import linalg as LA

import metod_alg as mt
from metod_alg import objective_functions as mt_obj
from metod_alg import metod_algorithm_functions as mt_alg


def plot_boxplots(arr1, obj, seed, sigma_sq_list):
      plt.figure(figsize=(5, 5))
      plt.boxplot(arr1)
      plt.xticks(np.arange(1, 4), sigma_sq_list, rotation=90)
      plt.ylim(0, 8)
      plt.ylabel(r'$\| \| \nabla f(x_n^{(0)})})\|\|$', size=12)
      plt.savefig('boxplots_norm_grad_obj_%s_seed_%s_.pdf' 
                  % (obj, seed),
                  bbox_inches="tight")


if __name__ == "__main__":
      obj = str(sys.argv[1])
      seed = int(sys.argv[2])

      sigma_sq_list = [0.03, 0.05, 0.07]
      data = np.zeros((100, 3))
      index = 0
      for sigma_sq in sigma_sq_list:
            data[:, index] = np.genfromtxt('norm_grad_%s_d_2_rs_%s'
                                           '_sigma_sq_%s.csv' %
                                           (obj, seed, sigma_sq),
                                            delimiter=',')
            index += 1
      plot_boxplots(data, obj, seed, sigma_sq_list)
