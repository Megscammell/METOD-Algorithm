import numpy as np
import sys
import matplotlib.pyplot as plt; plt.rcdefaults()
from numpy import linalg as LA

import metod as mt
from metod import objective_functions as mt_obj
from metod import metod_algorithm_functions as mt_alg


def plot_functions_with_anti_gradient_its(obj, test_num, seed, num_p, met):
      np.random.seed(seed)
      d = 2
      P = 4
      lambda_1 = 1
      lambda_2 = 5
      if obj == 'quad':
            f = mt_obj.quad_function 
            g = mt_obj.quad_gradient
            store_x0, matrix_combined = (mt_obj.function_parameters_quad
                                        (P, d, lambda_1, lambda_2))
            store_x0 = np.array([[0.96, 0.09],
                                 [0.86, 0.9],
                                 [0.2, 0.98],
                                 [0.12, 0.22]])
            args = P, store_x0, matrix_combined

      elif obj == 'sog':
            f = mt_obj.sog_function 
            g = mt_obj.sog_gradient
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

      relax_sd_it = 1
      usage = 'metod_algorithm'
      tolerance=0.00001
      projection=False
      bound_1 = 0
      bound_2 = 1
      option='minimize'
      initial_guess=0.05

      for _ in range(num_p):
            x = np.random.uniform(0, 1, (d,))
            descended_x_points, its = (mt_alg.apply_sd_until_stopping_criteria
                                      (x, d, projection, tolerance, option,
                                       met, initial_guess, args, f, g, 
                                       bound_1, bound_2, usage, relax_sd_it))

            chosen_x1 = descended_x_points[0:descended_x_points.shape[0]][:,0]
            chosen_x2 = descended_x_points[0:descended_x_points.shape[0]][:,1]
            
            plt.scatter(chosen_x1, chosen_x2, s=20, color='blue')
            plt.plot(chosen_x1, chosen_x2, 'blue')

      plt.contour(X, Y, Z, 50, cmap='RdGy', alpha=0.5)
      if obj == 'quad':
            plt.savefig('anti_grad_its_%s_d=2_rs_%s.pdf' % (obj, seed))
      elif obj == 'sog': 
            norm_grad = np.zeros((100))
            for k in range(100):
                  x = np.random.uniform(0, 1, (d,))
                  norm_grad[k] = LA.norm(g(x, *args))
            
            np.savetxt("norm_grad_%s_d_2_rs_%s_sigma_sq_%s.csv" %
                 (obj, seed, sigma_sq), norm_grad, delimiter=",")
            plt.savefig('anti_grad_its_%s_d=2_rs_%s_sigma_sq_%s.pdf'
                         % (obj, seed, sigma_sq))

if __name__ == "__main__":
    obj = str(sys.argv[1])
    test_num = int(sys.argv[2])
    seed = int(sys.argv[3])
    num_p = int(sys.argv[4])
    met = str(sys.argv[5])
    plot_functions_with_anti_gradient_its(obj, test_num, seed, num_p, met)