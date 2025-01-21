import numpy as np

from src.fitness_function import basic_fitness_function
from src.greedy_optimizations import calculate_delta_cost


def test_calculate_delta_cost():
    flow = np.array([[0, 10, 1, 1], [10, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    distance = np.array([[0, 5, 1, 1], [5, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    route = np.array([0, 1, 2, 3])
    i = 1
    j = 2

    current_cost = basic_fitness_function(flow, distance, route)
    new_route = route.copy()
    new_route[i], new_route[j] = route[j], route[i]

    computed_delta = calculate_delta_cost(flow, distance, route, i, j)
    expected_delta = basic_fitness_function(flow, distance, new_route) - current_cost

    assert computed_delta == expected_delta


if __name__ == '__main__':
    test_calculate_delta_cost()
