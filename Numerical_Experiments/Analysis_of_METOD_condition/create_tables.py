import numpy as np
import sys
import pandas as pd


def compute_tables(beta_list, m_list, func_name, d, projection, relax_sd,
                   num, met):
    """
    Compute and save summary tables of results for points belonging to the same
    and different regions of attraction.

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
    projection : boolean
                 If projection is True, points are projected back to
                 bounds_set_x. If projection is False, points are
                 kept the same.
    relax_sd_it : float or integer
                  Multiply the step size by a small constant in [0, 2], to
                  obtain a new step size for steepest descent iterations. This
                  process is known as relaxed steepest descent [2].
    num : integer
          Iteration number to start comparing the METOD algorithm inequality
          [1, Eq. 4].
    met : string
           If option = 'minimize' or option = 'minimize_scalar', choose
           appropiate method. For more information see
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize.html#scipy.optimize.minimize
           - https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)
    2) Raydan, M., Svaiter, B.F.: Relaxed steepest descent and
       cauchy-barzilai- borwein method. Computational Optimization and
       Applications 21(2), 155–167 (2002)
    """
    store_data_sm = np.zeros((len(beta_list) * len(m_list), len(m_list)))
    store_data_nsm = np.zeros((len(beta_list) * len(m_list), len(m_list)))
    index = len(m_list)
    for beta in beta_list:
        tot_sm = np.genfromtxt('%s_beta=%s_tot_sm_d=%s_%s_relax_c=%s'
                               '_num=%s_%s.csv' %
                               (func_name, beta, d, projection, relax_sd,
                                num, met),
                               delimiter=',')

        tot_nsm = np.genfromtxt('%s_beta=%s_tot_nsm_d=%s_%s_relax_c=%s'
                                '_num=%s_%s.csv' %
                                (func_name, beta, d, projection, relax_sd,
                                 num, met),
                                delimiter=',')

        fails_sm = np.genfromtxt('%s_beta=%s_sm_d=%s_%s_relax_c=%s'
                                 '_num=%s_%s.csv' %
                                 (func_name, beta, d, projection, relax_sd,
                                  num, met),
                                 delimiter=',')

        fails_nsm = np.genfromtxt('%s_beta=%s_nsm_d=%s_%s_relax_c=%s'
                                  '_num=%s_%s.csv' %
                                  (func_name, beta, d, projection, relax_sd,
                                   num, met),
                                  delimiter=',')

        if np.all(tot_sm > 0):
            prop_sm = np.genfromtxt('%s_beta=%s_sm_d=%s_prop_%s_relax_c=%s'
                                    '_num=%s_%s.csv' %
                                    (func_name, beta, d, projection, relax_sd,
                                     num, met),
                                    delimiter=',')
            assert(np.all((fails_sm / tot_sm)[:11, :11] == prop_sm))
            store_data_sm[index-len(m_list):index] = np.round(prop_sm, 4)[:len(m_list), :len(m_list)]

        if np.all(tot_nsm > 0):
            prop_nsm = np.genfromtxt('%s_beta=%s_nsm_d=%s_prop_%s_relax_c=%s'
                                     '_num=%s_%s.csv' %
                                     (func_name, beta, d, projection, relax_sd,
                                      num, met),
                                     delimiter=',')
            assert(np.all((fails_nsm / tot_nsm)[:11, :11] == prop_nsm))
            store_data_nsm[index-len(m_list):index] = np.round(prop_nsm, 4)[:len(m_list), :len(m_list)]
        index += len(m_list)

    list_pd = [x for b in beta_list for x in (m_list)]
    if np.all(tot_sm > 0):
        data_table_pd_sm = pd.DataFrame(data=store_data_sm,
                                        index=list_pd,
                                        columns=m_list)
        data_table_pd_sm.to_csv('%s_data_sm_tot_d=%s_%s_relax_c=%s'
                                '_num=%s_%s.csv' %
                                (func_name, d, projection,
                                 relax_sd, num, met))
        with open('%s_data_sm_tot_d=%s_%s_relax_c=%s_num=%s_%s.tex'
                  % (func_name, d, projection, relax_sd, num, met), 'w') as tf:
            tf.write(data_table_pd_sm.to_latex())

    if np.all(tot_nsm > 0):
        data_table_pd_nsm = pd.DataFrame(data=store_data_nsm,
                                         index=list_pd,
                                         columns=(m_list))
        data_table_pd_nsm.to_csv('%s_data_nsm_tot_d=%s_%s_relax_c=%s'
                                 '_num=%s_%s.csv' %
                                 (func_name, d, projection, relax_sd,
                                  num, met))
        with open('%s_data_nsm_tot_d=%s_%s_relax_c=%s_num=%s_%s.tex'
                  % (func_name, d, projection, relax_sd, num, met), 'w') as tf:
            tf.write(data_table_pd_nsm.to_latex())


if __name__ == "__main__":
    func_name = str(sys.argv[1])
    projection = False
    relax_sd = 1
    met = 'Brent'
    if func_name == 'quad':
        beta_list = [0.01, 0.1, 0.2]
        m_list = [0, 1, 2]
        d = 100
        num = 0
    elif func_name == 'sog':
        beta_list = [0.01, 0.1, 0.2]
        m_list = [1, 2]
        d = 20
        num = 1
    elif func_name == 'styb':
        beta_list = [0.001, 0.005, 0.01, 0.05]
        m_list = [0, 1]
        d = 5
        num = 0
    elif func_name == 'qing':
        beta_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        m_list = [1, 2, 3]
        d = 5
        num = 0
    elif func_name == 'zak':
        beta_list = [0.00000001, 0.000001, 0.0001]
        m_list = [0, 1]
        d = 10
        num = 0
    elif func_name == 'shekel':
        beta_list = [0.01, 0.1, 0.2]
        m_list = [1, 2]
        d = 4
        num = 1
    elif func_name == 'hart':
        beta_list = [0.001, 0.01, 0.1]
        m_list = [1, 2]
        d = 6
        num = 1
    compute_tables(beta_list, m_list, func_name, d, projection, relax_sd,
                   num, met)
