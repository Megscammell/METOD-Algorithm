import numpy as np
import sys
import matplotlib.pyplot as plt; plt.rcdefaults()

import metod_alg as mt
from metod_alg import objective_functions as mt_obj


def plot_functions(obj, test_num, seed):
      np.random.seed(seed)
      d = 2
      P = 4
      lambda_1 = 1
      lambda_2 = 5
      if obj == 'quad':
            f = mt_obj.several_quad_function 
            store_x0, matrix_combined = (mt_obj.function_parameters_several_quad(P, d, lambda_1, lambda_2))
            store_x0 = np.array([[0.96, 0.09],
                                 [0.86, 0.9],
                                 [0.2, 0.98],
                                 [0.12, 0.22]])
            args = P, store_x0, matrix_combined

      elif obj == 'sog':
            f = mt_obj.sog_function 
            store_x0, matrix_combined, store_c = (mt_obj.function_parameters_sog(P, d, lambda_1, lambda_2))

            store_x0 = np.array([[0.96, 0.09],
                                 [0.86, 0.9],
                                 [0.2, 0.98],
                                 [0.12, 0.22]])
            store_c = np.array([0.8 ,0.7, 0.9, 0.75])
            sigma_sq = 0.05
            args = P, sigma_sq, store_x0, matrix_combined, store_c

      x = np.linspace(0, 1, test_num)
      y = np.linspace(0, 1, test_num)
      Z = np.zeros((test_num, test_num))
      X, Y = np.meshgrid(x, y)
      for i in range(test_num):
            for j in range(test_num):
                  x1_var = X[i, j]
                  x2_var = Y[i, j]
                  Z[i, j] = f(np.array([x1_var, x2_var]).reshape(2, ), *args)

      plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.5)
      plt.savefig('%s_d=2_rs_%s.pdf' % (obj, seed))


if __name__ == "__main__":
    obj = str(sys.argv[1])
    test_num = int(sys.argv[2])
    seed = int(sys.argv[3])
    plot_functions(obj, test_num, seed)