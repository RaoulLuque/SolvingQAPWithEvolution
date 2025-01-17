from unittest.mock import patch

import numpy as np

from src.selection import tournament_selection_two_tournament, tournament_selection_two_tournament_bulk


def test_tournament_selection_two_tournament():
    population = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    population_fitness = np.array([10, 1])
    expected_selected_chromosomes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    mocked_random_choice = np.array([0, 1])
    mocked_rand = 0.5
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        with patch("numpy.random.rand", return_value=mocked_rand):
            selected_chromosomes = tournament_selection_two_tournament(population, population_fitness)
    assert np.array_equal(selected_chromosomes, expected_selected_chromosomes)

    population = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    population_fitness = np.array([1, 10])
    expected_selected_chromosomes = np.array([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        with patch("numpy.random.rand", return_value=mocked_rand):
            selected_chromosomes = tournament_selection_two_tournament(population, population_fitness)
    assert np.array_equal(selected_chromosomes, expected_selected_chromosomes)


def test_tournament_selection_two_tournament_bulk():
    population = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    population_fitness = np.array([10, 1])
    expected_selected_chromosomes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    mocked_random_choice = np.array([0, 1, 0, 1])
    mocked_rand = np.array([0.5, 0.5])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        with patch("numpy.random.rand", return_value=mocked_rand):
            selected_chromosomes = tournament_selection_two_tournament_bulk(population, population_fitness)
    assert np.array_equal(selected_chromosomes, expected_selected_chromosomes)

    population = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    population_fitness = np.array([1, 10])
    expected_selected_chromosomes = np.array([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    with patch("numpy.random.choice", return_value=mocked_random_choice):
        with patch("numpy.random.rand", return_value=mocked_rand):
            selected_chromosomes = tournament_selection_two_tournament_bulk(population, population_fitness)
    assert np.array_equal(selected_chromosomes, expected_selected_chromosomes)
