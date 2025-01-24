import numpy as np
from numpy import ndarray


def apply_mutation_to_population(population: ndarray, mutation_function: callable, mutation_rate: float) -> ndarray:
    """
    Apply a mutation function to a population with a given mutation rate.
    :param population: The population to mutate
    :param mutation_function: The mutation function to apply
    :param mutation_rate: The mutation rate
    :return: The mutated population
    """
    for chromosome_index in range(population.shape[0]):
        if np.random.rand() < mutation_rate:
            population[chromosome_index] = mutation_function(population[chromosome_index])
    return population


def swap_mutation(chromosome: ndarray) -> ndarray:
    """
    Perform a swap mutation on the chromosome.
    :param chromosome: The chromosome to mutate
    :return: The mutated chromosome
    """
    index1, index2 = np.random.choice(chromosome.shape[0], 2, replace=False)
    chromosome_entry_one = chromosome[index1]
    chromosome[index1] = chromosome[index2]
    chromosome[index2] = chromosome_entry_one
    return chromosome
