import numpy as np


def bulk_basic_fitness_function(flow_matrix: np.ndarray, distance_matrix: np.ndarray, chromosomes: np.ndarray) -> np.ndarray:
    """
    Calculate the fitness value of multiple chromosomes at once.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosomes: Three-dimensional numpy array representing the chromosomes
    :return: One-dimensional numpy array containing the fitness values of the chromosomes
    """
    return np.sum(flow_matrix[np.newaxis, :, :] * distance_matrix[chromosomes[:, :, np.newaxis], chromosomes[:, np.newaxis, :]],
                  axis=(1, 2))
