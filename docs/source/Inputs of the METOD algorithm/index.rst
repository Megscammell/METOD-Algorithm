.. role:: bash(code)
   :language: bash

Inputs of the METOD algorithm
===========================================

In this section, details on the required inputs and optional inputs of the METOD Algorithm are provided.

Required Inputs
----------------------

The required inputs of :bash:`metod.py` are listed below, along with the variable type. All required inputs need to be updated before running the METOD Algorithm. 


.. list-table::
   :widths: 10 10 50
   :header-rows: 1

   * - Input parameter
     - Type
     - Description
   * - :bash:`f`
     - function
     - Function evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a float.
   * - :bash:`g`
     - function
     - Gradient evaluated at a point, which is a 1-D array with shape :bash:`(d, )`, and outputs a 1-D array with shape :bash:`(d, )`.
   * - :bash:`func_args`
     - tuple
     - Any function parameters needed to compute :bash:`f` or :bash:`g`.
   * - :bash:`d`
     - integer
     - Size of dimension.



Optional Inputs
----------------------

The optional inputs of :bash:`metod.py` are listed below, along with the variable type.


.. list-table::
   :widths: 10 10 12 45
   :header-rows: 1

   * - Input parameter name
     - Default input
     - Type
     - Description
   * - :bash:`num_points`
     - :bash:`1000`
     - integer
     - The number of points generated before stopping the METOD Algorithm. 
   * - :bash:`beta`
     - :bash:`0.01`
     - float
     - Small constant step size :math:`\beta` to compute the partner points :math:`\tilde {x}_n` of :math:`x_n` (see :eq:`sd1`). It is required that :math:`\beta < 1 / \lambda_{max}`.
   * - :bash:`tolerance`
     - :bash:`0.00001`
     - float
     - Stopping condition for anti-gradient descent iterations. That is, apply anti-gradient descent iterations until :math:`\| \nabla f(x_n^{(k)}) \| < \delta`, where :math:`\delta` is the tolerance. If :math:`\| \nabla f(x_n^{(0)}) \| < \delta`, another starting point :math:`x_n^{(0)}` is used. To avoid this, it is recommended to choose suitable function parameters and dimension. 
   * - :bash:`projection`
     - :bash:`False`
     - boolean
     - If :bash:`projection = True`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is projected into a feasible domain :math:`\mathfrak{X}`. If :bash:`projection = False`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is not projected.
   * - :bash:`const`
     - :bash:`0.1`
     - float
     - Value of :math:`\eta` used in :eq:`sd3`.
   * - :bash:`m`
     - :bash:`3`
     - integer
     - The number of iterations of anti-gradient descent to apply to a point :math:`x_n` before making decision on terminating descents (See :ref:`Step 2 of the METOD Algorithm <metodalg>`). 
   * - :bash:`option`
     - :bash:`‘minimize_scalar’`
     - string
     -  Option of algorithm in Python to compute :math:`\gamma_n^{(k)}` for anti-gradient descent iterations :eq:`sd`. Choose from :bash:`option = ‘minimize’` or :bash:`option = ‘minimize_scalar’`.
        
        See :cite:`2020SciPy-NMeth` for more details on scipy.optmize.minimize and scipy.optmize.minimize_scalar.
   * - :bash:`met`
     - :bash:`‘Brent’`
     - string
     - A method is required for :bash:`option = ‘minimize’` or :bash:`option = ‘minimize_scalar’` (see :cite:`2020SciPy-NMeth` for more details).
   * - :bash:`initial_guess`
     - :bash:`0.005`
     - float
     - Initial guess passed to :bash:`option = ‘minimize’` and the upper bound for the bracket interval when :bash:`option = ‘minimize_scalar’` for :bash:`met = ‘Brent’` and :bash:`met = ‘Golden’`.
   * - :bash:`set_x`
     - :bash:`‘sobol’`
     - string
     - If  :bash:`set_x = ‘random’`, then :math:`x_n^{(0)} \in \mathfrak{X}` :math:`(n=1,...,N)` is generated uniformly at random for the METOD Algorithm. If  :bash:`set_x = ‘sobol’`, then a 2-D array with shape :bash:`(num_points * 2, d)` of Sobol sequence samples are generated using SALib :cite:`herman2017salib`. We transform the Sobol sequence samples so that samples are within :math:`\mathfrak{X}`. The Sobol sequence samples are then shuffled at random and selected by the METOD Algorithm.
   * - :bash:`bounds_set_x`
     - :bash:`(0,1)`
     - tuple
     - Feasible domian :math:`\mathfrak{X}`.
   * - :bash:`relax_sd_it`
     - :bash:`1`
     - float or integer
     - Multiply the step size :math:`\gamma_n^{(k)}` by a small constant in [0, 2], to obtain a new step size for anti-gradient descent iterations. This process is known as relaxed steepest descent :cite:`raydan2002relaxed`.

