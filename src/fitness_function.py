import numpy as np


def basic_fitness_function(flow_matrix: np.ndarray, distance_matrix: np.ndarray, chromosome: np.ndarray) -> int:
    return np.sum(np.diag(flow_matrix.dot(chromosome).dot(distance_matrix).dot(chromosome.T)))
