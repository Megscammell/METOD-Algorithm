import numpy as np
from hypothesis import given, settings, strategies as st

from metod_alg import prev_metod_algorithm as prev_mt_alg


def test_1():
       """
       Check that the closest minimizer is np.array([0.45, 0.65, 0.75, 0.9]).
       """
       possible_region_numbers = [0, 3]
       x_2 = np.array([0.5, 0.7, 0.8, 0.9])
       discovered_minimizers = np.array([[0.8, 0.7, 0.9, 0.2],
                                         [0, 0, 0, 0],
                                         [0.1, 0.3, 0.2, 0.7],
                                         [0.45, 0.65, 0.75, 0.9]])
       c = prev_mt_alg.regions_greater_than_2(possible_region_numbers,
                                              discovered_minimizers, x_2)
       assert(c == 3)


def test_2():
       """
       Check that the closest minimizer is np.array([0.45, 0.65, 0.75, 0.9]).
       """
       possible_region_numbers = [0, 1, 3]
       x_2 = np.array([0.5, 0.7, 0.8, 0.9])
       discovered_minimizers = np.array([[0.8, 0.7, 0.9, 0.2],
                                         [0.45, 0.65, 0.75, 0.9],
                                         [0, 0, 0, 0],
                                         [0.1, 0.3, 0.2, 0.7]])
       c = prev_mt_alg.regions_greater_than_2(possible_region_numbers,
                                              discovered_minimizers, x_2)
       assert(c == 1)