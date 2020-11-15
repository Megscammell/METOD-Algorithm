.. role:: bash(code)
   :language: bash

Optional Inputs
=================

The optional inputs of :bash:`metod.py` are listed below, along with the variable type.


.. _numpoints:

:bash:`num_points` (integer)
-------------------------------

The number of random points generated before stopping the METOD algorithm. It is recommended to set :bash:`num_points` as a large value in order to identify as many local minima as possible. However, this will increase the run time of the METOD algorithm. 

The default value is ::

    num_points = 1000 

.. _beta:

:bash:`beta` (float)
----------------------

Small constant step size :math:`\beta` to compute the partner points :math:`\tilde {x_n}` of :math:`x_n` (see :eq:`sd1`). The value of beta must be strictly smaller than 1.

The default value is ::

    beta = 0.01

.. _tol:

:bash:`tolerance` (integer or float)
--------------------------------------

Stopping condition for anti-gradient descent iterations. That is, apply anti-gradient descent iterations until :math:`\| \nabla f(x_n^{(k)}) \| < \delta`, where :math:`\delta` is the tolerance. To apply the METOD algorithm, must set ::

    usage = ‘metod_algorithm’

Then the default is ::
    
    tolerance = 0.00001. 

.. _proj:

:bash:`projection` (boolean)
-------------------------------

Sometimes :math:`x_n^{(k+1)}` may not be contained within specified bounds (i.e :math:`[0, 1]`). Hence, we can project :math:`x_n^{(k+1)}` to the specified bounds. The default is ::

    projection = False.

This will allow :math:`x_n^{(k+1)}` to remain outside specified bounds.

.. _const:

:bash:`const` (float)
----------------------

Value of :math:`\eta` used in :eq:`sd3`. We should have that :math:`\eta` is not too small as this may classify two local minima as belonging to different regions of attraction Also, :math:`\eta` should not be too large, as this may classify two local minima as belonging to the same region of attraction.

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

There are various methods to choose from when applying scipy.optmize.minimize or scipy.optmize.minimize\_scalar. It can be shown that scipy.optmize.minimize using the L-BFGS-B method (see :cite:`byrd1995limited`, :cite:`zhu1997algorithm`, :cite:`morales2011remark`) computes :math:`\gamma_n^{(k)}` in the fastest amount of time when the minimum of several quadratic forms function is used. However, it is possible that many iterations of anti-gradient descent will need to be computed before some stopping criterion is met. This can be due to a number of reasons including, if :bash:`tolerance` is set to a very small value or if the objective function is complex. The METOD algorithm returns an error message if the number of iterations exceeds 200. This suggests that  :math:`\gamma_n^{(k)}` may not be computed accurately by the chosen :bash:`met` or that :bash:`tolerance` is too small. For the Sum of Gaussians objective function, sometimes an error message can be observed when the L-BFGS-B method when is used. This is due to :math:`\gamma_n^{(k)}` not being computed accurately by the L-BFGS-B method. Hence, the default is ::

    met = ‘Nelder-Mead’.

.. _ig:

:bash:`initial_guess` (float)
------------------------------

The scipy.optimize.minimize option requires an initial guess to be input by the user. This is recommended to be small, as :math:`\gamma_n^{(k)}` is the step size. The default is ::

    initial_guess = 0.05. 

.. _set:

:bash:`set_x` (numpy.random distribution, list or numpy.array)
----------------------------------------------------------------

If numpy.random distribution is selected, random starting points from :bash:`bounds_set_x` are generated for the METOD algorithm. If a list or a numpy array of length :bash:`num_points` is given, then the METOD algorithm uses each point in the list or numpy array as staring points. 

The default is ::

    set_x = np.random.uniform.

.. _bounds:

:bash:`bounds_set_x` (tuple)
-----------------------------------

Bounds for numpy.random distribution. The Default is ::

    bounds_set_x = (0, 1).

.. _nitc:

:bash:`no_inequals_to_compare` (string)
-----------------------------------------

Evaluate :eq:`sd2` with all iterations :math:`i=(M-1,...,K_l)` (‘All’) or two iterations (‘Two’) :math:`i=(M-1,M)`.

The default is ::

    no_inequals_to_compare = ‘All’.

.. _use:

:bash:`usage` (string)
-----------------------

Decide stopping criterion for anti-gradient descent iterations. Should always be set to ::

    usage = ‘metod_algorithm’

.. _relax:

:bash:`relax_sd_it` (integer or float)
----------------------------------------

Small constant in [0, 2] to multiply the step size :math:`\gamma_n^{(k)}` by for a anti-gradient descent iteration. This process is known as relaxed steepest descent :cite:`raydan2002relaxed`. The default is ::

    relax_sd_it = 1.

Bibliography
-------------

.. bibliography:: references.bib
   :style: plain
