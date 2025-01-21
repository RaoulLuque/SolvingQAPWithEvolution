import numpy as np
from numpy import ndarray


def basic_fitness_function(flow_matrix: ndarray, distance_matrix: ndarray, chromosome: ndarray) -> float:
    """
    Calculate the fitness value of a chromosome.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosome: One-dimensional numpy array representing the chromosome
    :return: The fitness value of the chromosome
    """
    return np.sum(flow_matrix * distance_matrix[chromosome[:, np.newaxis], chromosome[np.newaxis, :]])


def bulk_basic_fitness_function(flow_matrix: ndarray, distance_matrix: ndarray, chromosomes: ndarray) -> ndarray:
    """
    Calculate the fitness value of multiple chromosomes at once.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosomes: Three-dimensional numpy array representing the chromosomes
    :return: One-dimensional numpy array containing the fitness values of the chromosomes
    """
    return np.sum(flow_matrix[np.newaxis, :, :] * distance_matrix[chromosomes[:, :, np.newaxis], chromosomes[:, np.newaxis, :]],
                  axis=(1, 2))
