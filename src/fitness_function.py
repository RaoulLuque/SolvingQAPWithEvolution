import numpy as np


def basic_fitness_function(flow_matrix: np.ndarray, distance_matrix: np.ndarray, chromosome: np.ndarray) -> int:
    """
    Basic fitness function that calculates the cost of a given chromosome using the [formula from Wikipedia](https://en.wikipedia.org/wiki/Quadratic_assignment_problem#Formal_mathematical_definition)
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosome: Two-dimensional numpy array representing the chromosome (permutation of the identity matrix)
    :return: The cost of the given chromosome
    """
    return np.sum(np.diag(flow_matrix.dot(chromosome).dot(distance_matrix).dot(chromosome.T)))
