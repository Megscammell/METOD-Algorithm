.. role:: bash(code)
   :language: bash

Optional Inputs
=================

The optional inputs of :bash:`metod.py` are listed below, along with the variable type.


.. list-table::
   :widths: 10 10 50
   :header-rows: 1

   * - Input parameter
     - Type
     - Description
   * - :bash:`num_points`
     - integer
     - The number of points generated before stopping the METOD algorithm. The default value is :bash:`num_points = 1000`. 
   * - :bash:`beta`
     - float
     - Small constant step size :math:`\beta` to compute the partner points :math:`\tilde {x}_n` of :math:`x_n` (see :eq:`sd1`). It is required that :math:`\beta < 1 / \lambda_{max}`. The default value is :bash:`beta = 0.01`.
   * - :bash:`tolerance`
     - float or integer
     - Stopping condition for anti-gradient descent iterations. That is, apply anti-gradient descent iterations until :math:`\| \nabla f(x_n^{(k)}) \| < \delta`, where :math:`\delta` is the tolerance. If :math:`\| \nabla f(x_n^{(0)}) \| < \delta`, another starting point :math:`x_n^{(0)}` is used. To avoid this, it is recommended to choose suitable function parameters and dimension. The default is :bash:`tolerance = 0.00001`.
   * - :bash:`projection`
     - boolean
     - If :bash:`projection = True`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is projected into a feasible domain :math:`\mathfrak{X}`, where bounds for :math:`\mathfrak{X}` are given by :bash:`bounds_set_x`. If :bash:`projection = False`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is not projected. The default is :bash:`projection = False`.
   * - :bash:`const`
     - float
     - Value of :math:`\eta` used in :eq:`sd3`. The default is :bash:`const = 0.1`.
   * - :bash:`m`
     - integer
     - The number of iterations of anti-gradient descent to apply to a point :math:`x_n` before making decision on terminating descents (See :ref:`Step 2 of the METOD algorithm <metodalg>`). The default value is :bash:`m = 3`.
   * - :bash:`option`
     - string
     - Choose from :bash:`option = ‘minimize’`, :bash:`option = ‘minimize_scalar’` or :bash:`option = ‘forward_backward_tracking’`. See :cite:`2020SciPy-NMeth` for more details on scipy.optmize.minimize and scipy.optmize.minimize_scalar. The default is :bash:`option = ‘minimize_scalar’`.
   * - :bash:`met`
     - string
     - If :bash:`option = ‘minimize’` or :bash:`option = ‘minimize_scalar’`, a method is required. If :bash:`option = ‘forward_backward_tracking’`, a method is not required. As the default is :bash:`option = ‘minimize_scalar’`, the default is :bash:`met = ‘Brent’`.
   * - :bash:`initial_guess`
     - float
     - Initial guess passed to :bash:`option = ‘minimize’`, :bash:`option = ‘forward_backward_tracking’` and the upper bound for the bracket interval when either :bash:`met = ‘Brent’` or :bash:`met = ‘Golden’` for :bash:`option = ‘minimize_scalar’`. The default is :bash:`initial_guess = 0.005`.
   * - :bash:`set_x`
     - string
     - If  :bash:`set_x = ‘random’`, then :math:`x_n^{(0)} \in \mathfrak{X}` :math:`(n=1,...,N)` is generated uniformly at random for the METOD algorithm, where :math:`\mathfrak{X}` is given by :bash:`bounds_set_x`. If  :bash:`set_x = ‘sobol’`, then a :bash:`numpy.array` with shape :bash:`(num_points * 2, d)` of Sobol sequence samples are generated using SALib :cite:`herman2017salib`. We transform the Sobol sequence samples so that samples are within :math:`\mathfrak{X}`. The Sobol sequence samples are then shuffled at random and selected by the METOD algorithm. The default is :bash:`set_x = ‘sobol’`.
   * - :bash:`bounds_set_x`
     - tuple
     - Feasible domian :math:`\mathfrak{X}` used for :bash:`set_x = ‘random’`, :bash:`set_x = ‘sobol’` and also for :bash:`projection = True`. The default is :bash:`bounds_set_x = (0, 1)`.
   * - :bash:`relax_sd_it`
     - float or integer
     - Multiply the step size by a small constant in [0, 2], to obtain a new step size for anti-gradient descent iterations. This process is known as relaxed steepest descent :cite:`raydan2002relaxed`. The default is :bash:`relax_sd_it = 1`.

