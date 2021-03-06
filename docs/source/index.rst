Welcome to METOD Algorithm's documentation!
===========================================

Multistart is a global optimization technique and works by applying local descent to a number of starting points. Multistart can be inefficient, as local descent is applied to each starting point and the same local minimizers are discovered. The METOD (Multistart with Early Termination of Descents) algorithm can be more efficient than multistart, as some local descents are stopped early. This avoids repeated descents to the same local minimizer.

The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer, and often violates when the trajectories belong to different regions of attraction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   background
   Installation/index
   Tutorials/index
   How-to-guides/index
   References/index
   
