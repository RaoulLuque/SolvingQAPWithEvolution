from unittest.mock import patch

import numpy as np

from src.recombine import partially_mapped_crossover, order_crossing


def test_order_crossing():

        parent_one = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parent_two = np.array([7, 8, 9, 6, 5, 4, 3, 2, 1, 0])
        expected_child_one = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_child_two = np.array([7, 8, 9, 6, 5, 4, 3, 0, 1, 2])

        child_one, child_two = order_crossing(parent_one, parent_two)
        assert np.array_equal(child_one, expected_child_one)
        assert np.array_equal(child_two, expected_child_two)


def test_order_crossing_two():
    mocked_random_choice = np.array([3, 7])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        parent_one = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        parent_two = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])
        expected_child_one = np.array([3, 8, 2, 4, 5, 6, 7, 1, 9])
        expected_child_two = np.array([3, 4, 7, 8, 2, 6, 5, 9, 1])

        child_one, child_two = order_crossing(parent_one, parent_two)
        assert np.array_equal(child_one, expected_child_one)
        assert np.array_equal(child_two, expected_child_two)


def test_partially_mapped_crossover():
    mocked_random_choice = np.array([0, 7])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        parent_one = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parent_two = np.array([7, 8, 9, 6, 5, 4, 3, 2, 1, 0])
        expected_child_one = np.array([0, 1, 2, 3, 4, 5, 6, 9, 8, 7])
        expected_child_two = np.array([7, 8, 9, 6, 5, 4, 3, 0, 1, 2])

        child_one, child_two = partially_mapped_crossover(parent_one, parent_two)
        assert np.array_equal(child_one, expected_child_one)
        assert np.array_equal(child_two, expected_child_two)


def test_partially_mapped_crossover_two():
    mocked_random_choice = np.array([2, 7])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        parent_one = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parent_two = np.array([7, 8, 9, 6, 5, 4, 3, 2, 1, 0])
        expected_child_one = np.array([7, 8, 2, 3, 4, 5, 6, 9, 1, 0])
        expected_child_two = np.array([0, 1, 9, 6, 5, 4, 3, 7, 8, 2])

        child_one, child_two = partially_mapped_crossover(parent_one, parent_two)
        assert np.array_equal(child_one, expected_child_one)
        assert np.array_equal(child_two, expected_child_two)
