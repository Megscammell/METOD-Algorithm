.. highlight:: rst

.. _styled-numbered-lists:

Background
==========

Notation
---------
The following basic notation will be used throughout. 

* :math:`d`:  dimension;

* :math:`\mathfrak{X} \subset \mathbb{R}^d`: feasible domain;

* :math:`\lambda_{\max}`: maximal eigenvalue of a positive definite matrix;

* :math:`\| \cdot \|`: Euclidean norm of a vector in :math:`\mathbb{R}^d`;

* :math:`N`: total number of starting points;

* :math:`X = {x_1,x_2,...x_N}`: a sequence of points in :math:`\mathfrak{X}`;

* :math:`x_n=x_n^{(0)} \in X`, where :math:`n=(1,2, \ldots N)`: point chosen from :math:`{X}`;

* :math:`x_n^{(k)}`: the :math:`n`-th point after :math:`k` iterations of anti-gradient descent;

* :math:`x_n^{(K_n)}`: smallest :math:`k=K_n` such that :math:`\| \nabla f(x_n^{(k)}) \| < \delta`;

* :math:`f(x_n^{(k)})`:   objective function at :math:`x_n^{(k)}`;

* :math:`\nabla f(x_n^{(k)})`: gradient of :math:`f` at :math:`x_n^{(k)}`;

*  anti-gradient descent iteration:

.. math::
    :label: sd

    x_n^{(k+1)} = x_n^{(k)} - \gamma_n^{(k)} \nabla f(x_n^{(k)})

* :math:`\gamma_n^{(k)}`: step size for an anti-gradient descent iteration :eq:`sd`. Note that :math:`\gamma_n^{(k)} \geq 0`;

* partner point:

.. math::
    :label: sd1

    \tilde{x}_{n}^{(k)}= x_n^{(k)} - \beta \nabla f(x_n^{(k)})

* :math:`\beta` : small positive constant used as the step size for a parnter point :eq:`sd1`;

* :math:`\delta` and :math:`\eta`: small positive constants;
* :math:`M`: the minimum number of anti-gradient descent iterations :eq:`sd` applied at each initial point :math:`x_n=x_n^{(0)}` where :math:`n=(1,2, \ldots, N)`;
* :math:`l`: index for the local minimizers :math:`l=1,...,L`, where :math:`L` is the total number of local minimizers found so far;
* :math:`A_l`: :math:`l`-th region of attraction;
* :math:`x_l^{(K_l)}`: :math:`l`-th local minimizer.

Main Conditions of the METOD Algorithm
----------------------------------------

The first main condition of the METOD Algorithm tests the following condition for each :math:`l=(1,...,L)` and all :math:`i=(M-1,...,K_l)`

.. math::
    :label: sd2

    \| \tilde{x}_{n}^{(M)}- \tilde{x}_{l}^{(i)} \| <  \| {x}_{n}^{(M)}- {x}_{l}^{(i)} \| \ and \  \| \tilde{x}_{n}^{(M-1)}- \tilde{x}_{l}^{(i)} \| <  \| {x}_{n}^{(M-1)}- {x}_{l}^{(i)} \|

For a given :math:`l`, if condition :eq:`sd2` holds for all :math:`i=(M-1,...,
K_l)`, then it is possible :math:`x_n` belongs to :math:`A_l`, the same region 
of attraction corresponding to :math:`x_l^{(K_l)}`, and we terminate 
anti-gradient descent iterations :eq:`sd`. Let :math:`S_n` be the set of 
indices :math:`l`, such that :math:`x_n` may belong to region of attraction 
:math:`A_l`. If condition :eq:`sd2` holds, we add index :math:`l` to the set 
:math:`S_n`.

If condition :eq:`sd2` does not hold for a point :math:`x_n` with 
any :math:`l=(1,...,L)`, then we apply anti-gradient descent iterations 
:eq:`sd` until a minimizer :math:`x_n^{(K_n)}` is found. However, we may have that minimizer :math:`x_n^{(K_n)}` has already 
been discovered. As a consequence, the second main condition of the algorithm 
is to ensure that all discovered local minimizers are unique. To 
return unique local minimizers only, the following condition is tested for all 
:math:`i=(1,...,L)` and :math:`j=(i + 1,...,L)`.

.. math::
    :label: sd3

    \| {x}_{i}^{(K_i)}- {x}_{j}^{(K_j)} \| >  \eta

If condition :eq:`sd3` fails for any :math:`j`, then minimizers :math:`x_i^{
(K_i)}` and :math:`x_j^{(K_j)}` are the same and :math:`j` is 
removed from the set of indices :math:`l=(1,...,L)`.

.. _metodalg:

METOD Algorithm
-----------------

The METOD Algorithm can be split into the following three parts.

.. rst-class:: bignums

1) **Initialization**

    Choose :math:`x_1=x_1^{(0)} \in {X}`. Use iterations :eq:`sd` to find a minimizer :math:`x_1^{(K_1)}`. For all points :math:`x_1^{(k)}` computed in :eq:`sd` with :math:`k =(M-1, M, \ldots, K_1)` compute the associated partner points using :eq:`sd1` and set :math:`L \gets 1`.

2) **Terminate anti-gradient descent iterations for** \ :math:`n`\ **-th   point**.

    For :math:`n=2` to :math:`N`

       Choose :math:`x_n=x_n^{(0)} \in {X}`. Compute :math:`x_n^{(j)}` for :math:`j=(1, \ldots, M)` and the associated partner points using :eq:`sd1`.

       For :math:`l=1` to :math:`L`

          If condition :eq:`sd2` is satisfied for every :math:`i=(M-1,...,K_l)`
            
             :math:`S_n \gets l`.

          If  :math:`S_n` contains one or more indices :math:`l` 

             Terminate iterations :eq:`sd` which have started at :math:`x_n`.

          Else

             Let :math:`x_{L+1} = x_n` and continue iterations :eq:`sd` until a minimizer :math:`x_{L+1}^{(K_{L+1})}` is found.
             
             For all points :math:`x_{L+1}^{(k)}` :math:`(k =M-1, \ldots, K_{L+1})`, compute the associated partner points using :eq:`sd1`. Set :math:`L \gets L+1`.

3) **Return unique minimizers from Step 2.**

    For :math:`i=1` to :math:`L`

       For :math:`j=i+1` to :math:`L`

          If condition :eq:`sd3` is not satisfied for :math:`{x}_{i}^{(K_i)}` and :math:`{x}_{j}^{(K_j)}`

             Remove index :math:`j` from the set of indices :math:`l=(1,...,L)`.

Code Structure
---------------

The METOD Algorithm code can be found `here <https://github.com/Megscammell/METOD-Algorithm/tree/master/src/metod_alg>`_. The main program that executes the METOD Algorithm is metod.py and the following diagram shows the various programs that contribute to metod.py.

.. figure:: code_structure.gif 
    :width: 1000px
    :align: center
    :height: 300px
    :alt: Code structure
