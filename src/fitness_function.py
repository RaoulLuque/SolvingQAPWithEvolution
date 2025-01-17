import numpy as np


def basic_fitness_function(flow_matrix: np.ndarray, distance_matrix: np.ndarray, chromosome: np.ndarray) -> int:
    """
    Basic fitness function that calculates the cost of a given chromosome using the [formula from Wikipedia](https://en.wikipedia.org/wiki/Quadratic_assignment_problem#Formal_mathematical_definition)
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosome: Two-dimensional numpy array representing the chromosome (permutation of the identity matrix)
    :return: The cost of the given chromosome
    """
    return np.sum(np.diag(flow_matrix.dot(chromosome).dot(distance_matrix.T).dot(chromosome.T)))


def bulk_basic_fitness_function(flow_matrix: np.ndarray, distance_matrix: np.ndarray, chromosomes: np.ndarray) -> np.ndarray:
    """
    Calculate the fitness value of multiple chromosomes at once.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosomes: Three-dimensional numpy array representing the chromosomes
    :return: One-dimensional numpy array containing the fitness values of the chromosomes
    """
    return np.trace(np.matmul(np.matmul(np.matmul(flow_matrix, chromosomes), distance_matrix.T), chromosomes.transpose(0, 2, 1)), axis1=1, axis2=2)
