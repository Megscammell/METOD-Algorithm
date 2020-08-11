import numpy as np
import pandas as pd

if __name__ == "__main__":
   """
   Comparing the total number of local minima found by METOD for each method
   compared with the number found by applying multistart. Use
   quad_sd_minimize_met_BFGS_beta_0.1_m=1_d=100_p=100_All.csv as the total
   number of local minima found by each function for multistart as the
   total number of local minima is same for each method.
   """
   d = 100
   beta = 0.01
   m = 2
   p = 50
   m_list = [2, 3, 4]
   beta_list = [0.01, 0.1]
   met_list = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'SLSQP', 'CG',
               'TNC', 'COBYLA', 'Brent', 'Golden']
   store_difference_metod_mult = np.zeros((len(met_list), 6))
   store_time_metod = np.zeros((len(met_list), 6))
   df_multistart = pd.read_csv('quad_sd_minimize_met_BFGS_beta_0.1_m=1_d=%s'
                               '_p=%s_All.csv' % (d, p))
   min_sd = np.array(df_multistart['number_minimas_per_func_multistart'])
   total_min_sd = np.sum(min_sd)
   index_row = 0
   for met in met_list:
      index_col = 0
      for m in m_list:
         for beta in beta_list:
            df2 = pd.read_csv('quad_testing_minimize_met_%s_beta_%s_m=%s'
                              '_d=%s_p=%s_All.csv' % (met, beta, m, d, p))

            """
            Check to ensure that total number of local minima found by METOD 
            does not exceed total number of local minima found by multistart.
            """

            assert(np.all(min_sd - np.array(df2
                          ['number_minimas_per_func_metod']) >= 0))
            store_difference_metod_mult[index_row, index_col] = (np.sum(min_sd - np.array(df2['number_minimas_per_func_metod'])))/total_min_sd

            store_time_metod[index_row, index_col] = np.mean(df2['time_metod'])
            index_col += 1
      index_row += 1

      store_time_metod = np.array(store_time_metod).reshape(len(met_list),
                                                            len(m_list) *
                                                            len(beta_list))
      midx = pd.MultiIndex.from_product([m_list, beta_list])
      df = pd.DataFrame(store_difference_metod_mult,
                        index=met_list, columns=midx)
      df.to_csv(df.to_csv('minima_sd_metod_results_beta_d=%s_p=%s.csv'
                          % (d, p)))

      with open('sd_metod_results_d_50.tex', 'w') as tf:
         tf.write(df.round(3).to_latex())

      store_time_metod = np.array(store_time_metod).reshape(len(met_list),
                                                            len(m_list) *
                                                            len(beta_list))
      midx = pd.MultiIndex.from_product([m_list, beta_list])
      df_time = pd.DataFrame(store_time_metod, index=met_list, columns=midx)
      df_time.to_csv(df_time.to_csv('time_results_beta_d=%s_p=%s.csv'
                                    % (d, p)))

      ax = df_time.plot.bar(figsize=(10,5), width=0.8)
      ax.legend([r'$M=2$, $\beta=0.01$', r'$M=2$, $\beta=0.1$',
                 r'$M=3$, $\beta=0.01$', r'$M=3$, $\beta=0.1$',
                 r'$M=4$, $\beta=0.01$', r'$M=4$, $\beta=0.1$'],
                bbox_to_anchor=[1.21, 1.02], loc='upper right',
                prop={'size': 10})
      ax.set_ylabel("Average time (s)")
      fig = ax.get_figure()
      fig.savefig('times_d_%s_p_%s.pdf' % (d, p), bbox_inches="tight")
