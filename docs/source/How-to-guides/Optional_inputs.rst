Optional Inputs
=================

The optional inputs of metod.py are listed below, along with the variable type.

* :ref:`num_points (integer) <numpoints>`
* :ref:`beta (float) <beta>`
* :ref:`tolerance (integer or float) <tol>`
* :ref:`projection (boolean) <proj>`
* :ref:`const (float) <const>`
* :ref:`m (integer) <m>`
* :ref:`option (string) <opt>`
* :ref:`met (string) <met>`
* :ref:`initial_guess (float) <ig>`
* :ref:`set_x (numpy.random distribution, list or numpy.array) <set>`
* :ref:`bounds_set_x (tuple) <bounds>`
* :ref:`no_inequals_to_compare (string) <nitc>`
* :ref:`usage (string) <use>`
* :ref:`relax_sd_it (integer or float) <relax>`

.. _numpoints:

num_points (integer)
---------------------

The number of random points generated before stopping the METOD algorithm. It is recommended to set num_points as a large value in order to identify as many local minima as possible. However, this will increase the run time of the METOD algorithm. 

The default value is ::

    num_points = 1000 

.. _beta:

beta (float)
-------------

Small constant step size :math:`\beta` to compute the partner points :math:`\tilde {x_n}` of :math:`x_n` (see :eq:`sd1`). The value of beta must be strictly smaller than 1.

The default value is ::

    beta = 0.01

.. _tol:

tolerance (integer or float)
-----------------------------

Stopping condition for steepest descent iterations. That is, apply steepest descent iterations until :math:`\| \nabla f(x_n^{(k)}) \| < \delta`, where :math:`\delta` is the tolerance. To apply the METOD algorithm, must set ::

    usage = ‘metod_algorithm’

Then the default is ::
    
    tolerance = 0.00001. 

.. _proj:

projection (boolean)
----------------------

Sometimes :math:`x_n^{(k+1)}` may not be contained within specified bounds (i.e :math:`[0, 1]`). Hence, we can project :math:`x_n^{(k+1)}` to the specified bounds. The default is ::

    projection = False.

This will allow :math:`x_n^{(k+1)}` to remain outside specified bounds.

.. _const:

const (float)
----------------------

Value of :math:`\eta` used in :eq:`sd3`. We should have that :math:`\eta` is not too small as this may classify two local minima as belonging to different regions of attraction, even if this is not the case. Also, :math:`\eta` should not be too large, as this may classify two local minima as belonging to the same region of attraction even if this is not the case.

The default is ::

    const = 0.1

.. _m:

m (integer)
------------

The number of iterations of steepest descent to apply to a point before making decision on terminating descents (See :ref:`Step 2 of the METOD algorithm <metodalg>`). 

The default value is ::

    m = 3

.. _opt:

option (string)
----------------

Exact line search is used to compute the step size :math:`\gamma_n^{(k)}` for each anti-gradient descent iteration :eq:`sd`. That is, we find :math:`\gamma_n^{(k)}` which satisfies

.. math::
    :label: minimizefunc

    \gamma_n^{(k)} = \text{argmin}_{\gamma > 0} f(x_n^{(k)} - \gamma \nabla f(x_n^{(k)}))

In order to compute :eq:`minimizefunc` in Python, the Scipy library :cite:`2020SciPy-NMeth` is used. Specifically, scipy.optmize.minimize and scipy.optmize.minimize_scalar can be used. In order to choose either option, the user can specify ‘minimize’ or ‘minimize\_scalar’ for scipy.optmize.minimize or scipy.optmize.minimize\_scalar respectively. 

The default is ::

    option = ‘minimize'.

.. _met:

met (string)
-------------

There are various methods to choose from when applying scipy.optmize.minimize or scipy.optmize.minimize\_scalar. It can be shown that scipy.optmize.minimize using the L-BFGS-B method (see :cite:`byrd1995limited`, :cite:`zhu1997algorithm`, :cite:`morales2011remark`) computes :math:`\gamma_n^{(k)}` in the fastest amount of time. Hence, the default is ::

    met = ‘L-BFGS-B’.

.. _ig:

initial_guess (float)
----------------------

The scipy.optimize.minimize option requires an initial guess to be input by the user. This is recommended to be small, as :math:`\gamma_n^{(k)}` is the step size. The default is ::

    initial_guess = 0.05. 

.. _set:

set_x (numpy.random distribution, list or numpy.array)
-------------------------------------------------------

If numpy.random distribution is selected, random starting points from bounds_set_x are generated for the METOD algorithm. If a list or a numpy array of length num_points is given, then the METOD algorithm uses each point in the list or numpy array as staring points. 

The default is ::

    set_x = np.random.uniform.

.. _bounds:

bounds_set_x (tuple)
----------------------

Bounds for numpy.random distribution. The Default is ::

    bounds_set_x = (0, 1).

.. _nitc:

no_inequals_to_compare (string)
---------------------------------

Evaluate :eq:`sd2` with all iterations :math:`i=(M-1,...,K_l)` (‘All’) or two iterations (‘Two’) :math:`i=(M-1,M)`.

The default is ::

    no_inequals_to_compare = ‘All’.

.. _use:

usage (string)
---------------

Decide stopping criterion for steepest descent iterations. Should always be set to ::

    usage = ‘metod_algorithm’

.. _relax:

relax_sd_it (integer or float)
-------------------------------

Small constant in [0, 2] to multiply the step size :math:`\gamma_n^{(k)}` by for a steepest descent iteration. This process is known as relaxed steepest descent :cite:`raydan2002relaxed`. The default is ::

    relax_sd_it = 1.

Bibliography
-------------

.. bibliography:: references.bib
   :style: plain
