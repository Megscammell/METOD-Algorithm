Background
==========

Overview of the METOD Algorithm
-------------------------------

Multistart is a celebrated global optimization technique and works by applying local descent to a number of starting points. Multistart can be deemed as inefficient, as local descent is applied for each starting point. The METOD algorithm can be more efficient than multistart due to some local descents being stopped early. This avoids repeated descents to the same local minimizer(s).
The early termination of descents in METOD is achieved by means of a particular inequality which holds when trajectories are from the region of attraction of the same local minimizer, and often violates when the trajectories belong to different regions of attraction.

Code Structure
---------------

Input Parameters
-----------------

Output Parameters
------------------

**See Examples**