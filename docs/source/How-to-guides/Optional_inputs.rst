.. role:: bash(code)
   :language: bash

Optional Inputs
=================

The optional inputs of :bash:`metod.py` are listed below, along with the variable type.


.. _numpoints:

:bash:`num_points` (integer)
-------------------------------

The number of random points generated before stopping the METOD algorithm. It is recommended to set :bash:`num_points` to a large value in order to identify as many local minima as possible. However, this will increase the run time of the METOD algorithm. 

The default value is ::

    num_points = 1000 

.. _beta:

:bash:`beta` (float)
----------------------

Small constant step size :math:`\beta` to compute the partner points :math:`\tilde {x}_n` of :math:`x_n` (see :eq:`sd1`). It is required that :math:`\beta < 1 / \lambda_{max}`.

The default value is ::

    beta = 0.01

.. _tol:

:bash:`tolerance` (integer or float)
--------------------------------------

Stopping condition for anti-gradient descent iterations. That is, apply anti-gradient descent iterations until :math:`\| \nabla f(x_n^{(k)}) \| < \delta`, where :math:`\delta` is the tolerance.

The default is ::
    
    tolerance = 0.00001. 

.. _proj:

:bash:`projection` (boolean)
-------------------------------

If :bash:`projection = True`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is projected into a feasible domain :math:`\mathfrak{X}`, where bounds for :math:`\mathfrak{X}` are given by :bash:`bounds_set_x`. If :bash:`projection = False`, then :math:`x_n^{(k)}` :math:`(k=1,...,K_n)` is not projected. The default is :bash:`projection = False`.  The default is ::

    projection = False.

This will allow :math:`x_n^{(k+1)}` to remain outside :math:`\mathfrak{X}`.

.. _const:

:bash:`const` (float)
----------------------

Value of :math:`\eta` used in :eq:`sd3`. We should have that :math:`\eta` is not too small as this may classify the same two local minimizers as belonging to different regions of attraction. Also, :math:`\eta` should not be too large, as this may classify two different local minimizers as belonging to the same region of attraction.

The default is ::

    const = 0.1

.. _m:

:bash:`m` (integer)
-----------------------

The number of iterations of anti-gradient descent to apply to a point before making decision on terminating descents (See :ref:`Step 2 of the METOD algorithm <metodalg>`). 

The default value is ::

    m = 3

.. _opt:

:bash:`option` (string)
-------------------------

Exact line search is used to compute the step size :math:`\gamma_n^{(k)}` for each anti-gradient descent iteration :eq:`sd`. That is, we find :math:`\gamma_n^{(k)}` which satisfies

.. math::
    :label: minimizefunc

    \gamma_n^{(k)} = \text{argmin}_{\gamma > 0} f(x_n^{(k)} - \gamma \nabla f(x_n^{(k)}))

In order to compute :eq:`minimizefunc` in Python, the Scipy library :cite:`2020SciPy-NMeth` is used. Specifically, scipy.optmize.minimize and scipy.optmize.minimize_scalar can be used. In order to choose either option, the user can specify :bash:`‘minimize’` or :bash:`‘minimize_scalar’` for scipy.optmize.minimize or scipy.optmize.minimize\_scalar respectively. 

The default is ::

    option = ‘minimize'.

.. _met:

:bash:`met` (string)
-----------------------

There are various methods to choose from when applying scipy.optmize.minimize or scipy.optmize.minimize\_scalar.

The default is ::

    met = ‘Nelder-Mead’.

.. _ig:

:bash:`initial_guess` (float)
------------------------------

The scipy.optimize.minimize option requires an initial guess to be input by the user. This is recommended to be small, as :math:`\gamma_n^{(k)}` is the step size. 

The default is ::

    initial_guess = 0.05. 

.. _set:

Note that the initial guess will not be used if the option is set to :bash:`‘minimize_scalar’`.

:bash:`set_x` (:bash:`numpy.random.uniform`, :bash:`numpy.array` or :bash:`sobol_sequence.sample`)
---------------------------------------------------------------------------------------------------------

If  :bash:`numpy.random.uniform` is selected, then :math:`x_n^{(0)} \in \mathfrak{X}` :math:`(n=1,...,N)` is generated uniformly at random for the METOD algorithm, where :math:`\mathfrak{X}` is given by :bash:`bounds_set_x`. If a :bash:`numpy.array` with shape :bash:`(num_points, d)` is given, then the METOD algorithm uses each point in the :bash:`numpy.array` as :math:`x_n^{(0)}` :math:`(n=1,...,N)`. If :bash:`sobol_sequence.sample` is selected, then a :bash:`numpy.array` with shape :bash:`(num_points, d)` of Sobol sequence samples are generated using SALib :cite:`herman2017salib`. The default is  :bash:`set_x = sobol_sequence.sample`.

The default is ::

    set_x = sobol_sequence.sample.

.. _bounds:

:bash:`bounds_set_x` (tuple)
-----------------------------------

Bounds :math:`\mathfrak{X}` used for :bash:`numpy.random.uniform` and also for :bash:`projection=True`. Although bounds are not required when :bash:`projection=False` and :bash:`set_x` is a :bash:`numpy.array` or :bash:`set_x=sobol_sequence.sample`, the default is ::

    bounds_set_x = (0, 1).
 


This is to ensure code will run if :bash:`set_x=numpy.random.uniform` or if :bash:`projection=True`. Note that if :math:`\| \nabla f(x_n^{(0)}) \| < \delta`, then another starting point :math:`x_n^{(0)}` will be randomly sampled from :bash:`numpy.random.uniform` with :bash:`bounds_set_x`. To avoid this, it is recommended to choose suitable function parameters and dimension. 


.. _relax:

:bash:`relax_sd_it` (integer or float)
----------------------------------------

Multiply the step size by a small constant in [0, 2], to obtain a new step size for anti-gradient descent iterations. This process is known as relaxed steepest descent :cite:`raydan2002relaxed`. The default is ::

    relax_sd_it = 1.

Bibliography
-------------

.. bibliography:: references.bib
   :style: plain
