from typing import Callable

import numpy as np

from src.chromosome import random_with_fitness
from src.config import NUMBER_OF_FACILITIES


def generate_initial_population_with_fitness(
        population_size: int,
        fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], int],
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an initial population with fitness values using the provided fitness function.
    :param population_size: The desired population size
    :param fitness_function: Fitness function to be used
    :param flow_matrix: Used by the fitness function to determine fitness of an individual
    :param distance_matrix: Used by the fitness function to determine fitness of an individual
    :return: Two numpy arrays. The first contains the population and the second contains the fitness values of the individuals/chromosomes
    """
    chromosomes = np.empty((population_size, NUMBER_OF_FACILITIES, NUMBER_OF_FACILITIES))
    fitness_values = np.empty(population_size)

    for i in range(population_size):
        chromosome, fitness = random_with_fitness(flow_matrix, distance_matrix, fitness_function)
        chromosomes[i] = chromosome
        fitness_values[i] = fitness

    return chromosomes, fitness_values
