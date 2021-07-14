import numpy as np


def regions_greater_than_2(possible_region_numbers, discovered_minimizers,
                           x_2):
    """
    If [1, Eq. 9] of the METOD algorithm is satisfied for more than one
    previously identified regions of attractions, then the tested point
    belongs to the closest region of attraction of a local minimizer.

    Parameters
    ----------
    possible_region_numbers : list
                              Positions of the previously identified
                              regions of attraction in which a point may
                              belong to.
    discovered_minimizers : list
                            Previously identified minimizers of regions
                            of attractions.
    x_2 : 1-D array with shape (d,), where d is the dimension.
          Point with M iterations of steepest descent applied.

    Returns
    -------
    classification : integer
                     Classification of x_2.

    References
    ----------
    1) Zilinskas, A., Gillard, J., Scammell, M., Zhigljavsky, A.: Multistart
       with early termination of descents. Journal of Global Optimization pp.
       1–16 (2019)

    """
    distances_to_minima = np.zeros((len(possible_region_numbers)))
    for i in range(len(possible_region_numbers)):
        x_minima_test = discovered_minimizers[possible_region_numbers[i]]
        distances_to_minima[i] = np.linalg.norm(x_minima_test - x_2)
    classification = possible_region_numbers[np.argmin(distances_to_minima)]
    return classification
