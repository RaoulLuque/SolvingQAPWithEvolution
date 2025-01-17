from typing import Callable

from src.config import NUMBER_OF_FACILITIES

import numpy as np


def random_with_fitness(flow_matrix: np.ndarray, distance_matrix: np.ndarray, fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], int]) -> tuple[np.ndarray, int]:
    """
    Create a random chromosome with a fitness value.
    """
    chromosome = np.random.permutation(np.eye(NUMBER_OF_FACILITIES))
    fitness = fitness_function(flow_matrix, distance_matrix, chromosome)

    return chromosome, fitness

