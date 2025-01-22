import numpy as np
from numpy import ndarray

from src import fitness_function
from src.config import NUMBER_OF_ITERATIONS_FOR_OPT


def two_opt(flow_matrix: ndarray, distance_matrix: ndarray, route: ndarray) -> ndarray:
    """
    Perform a 2-opt optimization on a given route.
    :param route: One-dimensional numpy array representing a single route (chromosome).
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix.
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix.
    :return: Optimized route (chromosome) as a numpy array.
    """
    best_route = route.copy()
    n = len(route)
    improved = True
    number_of_iteration = 0
    while improved and number_of_iteration < NUMBER_OF_ITERATIONS_FOR_OPT:
        number_of_iteration += 1
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # new_route = best_route.copy()
                # new_route[i], new_route[j] = best_route[j], best_route[i]
                # cost_delta = fitness_function.basic_fitness_function(flow_matrix, distance_matrix, new_route) - fitness_function.basic_fitness_function(flow_matrix, distance_matrix, best_route)
                cost_delta = calculate_delta_cost_numpy(flow_matrix, distance_matrix, best_route, i, j)
                if cost_delta < 0:
                    # Perform the 2-opt swap
                    tmp = best_route[j]
                    best_route[j] = best_route[i]
                    best_route[i] = tmp
                    # print(f"Cost of best_route: {fitness_function.basic_fitness_function(flow_matrix, distance_matrix, best_route)}")
                    improved = True

    return best_route


def calculate_delta_cost(flow_matrix: ndarray, distance_matrix: ndarray, route: ndarray, i: int, j: int) -> float:
    n = len(route)
    if i == j:
        return 0

    # Get the indices of the elements to be swapped
    a, b = route[i], route[j]

    # Calculate the delta cost
    delta = 0
    for k in range(n):
        if k != i and k != j:
            c = route[k]
            delta += (flow_matrix[i, k] * (distance_matrix[b, c] - distance_matrix[a, c]) +
                      flow_matrix[j, k] * (distance_matrix[a, c] - distance_matrix[b, c]) +
                      flow_matrix[k, i] * (distance_matrix[c, b] - distance_matrix[c, a]) +
                      flow_matrix[k, j] * (distance_matrix[c, a] - distance_matrix[c, b]))

    return delta


def calculate_delta_cost_numpy(flow_matrix: ndarray, distance_matrix: ndarray, route: ndarray, i: int, j: int) -> float:
    if i == j:
        return 0

    # Get the indices of the elements to be swapped
    a, b = route[i], route[j]

    # Create masks to exclude the i-th and j-th elements
    mask = np.ones(len(route), dtype=bool)
    mask[[i, j]] = False

    # Get the relevant rows and columns
    flow_i = flow_matrix[i, mask]
    flow_j = flow_matrix[j, mask]
    flow_k_i = flow_matrix[mask, i]
    flow_k_j = flow_matrix[mask, j]

    distance_b = distance_matrix[b, route[mask]]
    distance_a = distance_matrix[a, route[mask]]
    distance_c_b = distance_matrix[route[mask], b]
    distance_c_a = distance_matrix[route[mask], a]

    # Calculate the delta cost
    delta = np.sum(flow_i * (distance_b - distance_a) +
                   flow_j * (distance_a - distance_b) +
                   flow_k_i * (distance_c_b - distance_c_a) +
                   flow_k_j * (distance_c_a - distance_c_b))

    return delta
