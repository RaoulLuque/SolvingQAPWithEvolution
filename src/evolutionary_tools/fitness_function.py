import numpy as np
from numpy import ndarray
from tqdm import tqdm

from src.evolutionary_tools.greedy_optimizations import two_opt


def basic_fitness_function(flow_matrix: ndarray, distance_matrix: ndarray, chromosome: ndarray) -> float:
    """
    Calculate the fitness value of a chromosome.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosome: One-dimensional numpy array representing the chromosome
    :return: The fitness value of the chromosome
    """
    return np.sum(flow_matrix * distance_matrix[chromosome[:, np.newaxis], chromosome[np.newaxis, :]])


def bulk_basic_fitness_function(flow_matrix: ndarray, distance_matrix: ndarray, chromosomes: ndarray, final: bool = False, generation: int = 0) -> tuple[ndarray, ndarray]:
    """
    Calculate the fitness value of multiple chromosomes at once.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix
    :param chromosomes: Three-dimensional numpy array representing the chromosomes
    :param final: Boolean indicating if this is the final fitness calculation. Unused for this function
    :return: One-dimensional numpy array containing the fitness values of the chromosomes
    """
    return chromosomes, np.sum(flow_matrix[np.newaxis, :, :] * distance_matrix[chromosomes[:, :, np.newaxis], chromosomes[:, np.newaxis, :]], axis=(1, 2))


def bulk_basic_fitness_function_baldwinian(flow_matrix: ndarray, distance_matrix: ndarray, chromosomes: ndarray, final: bool = False, generation: int = 0) -> tuple[ndarray, ndarray]:
    """
    Calculate the fitness value of multiple chromosomes using Baldwinian evolution with 2-opt.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix.
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix.
    :param chromosomes: Three-dimensional numpy array representing the chromosomes.
    :param final: Boolean indicating if this is the final fitness calculation. In case of final == True, the optimized chromosomes will be returned instead
    :return: One-dimensional numpy array containing the fitness values of the chromosomes.
    """
    optimized_routes = chromosomes.copy()

    progress_bar_range = enumerate(tqdm(chromosomes, desc=f"Generation: {generation}"))
    for index, chromosome in progress_bar_range:
        optimized_routes[index] = two_opt(flow_matrix, distance_matrix, chromosome)

    if final:
        return bulk_basic_fitness_function(flow_matrix, distance_matrix, optimized_routes)
    else:
        fitness_values = bulk_basic_fitness_function(flow_matrix, distance_matrix, optimized_routes)[1]
        return chromosomes, fitness_values


def bulk_basic_fitness_function_lamarckian(flow_matrix: ndarray, distance_matrix: ndarray, chromosomes: ndarray, final: bool = False, generation: int = 0) -> tuple[ndarray, ndarray]:
    """
    Calculate the fitness value of multiple chromosomes using Lamarckian evolution with 2-opt.
    :param flow_matrix: Two-dimensional numpy array representing the flow matrix.
    :param distance_matrix: Two-dimensional numpy array representing the distance matrix.
    :param chromosomes: Three-dimensional numpy array representing the chromosomes.
    :param final: Boolean indicating if this is the final fitness calculation. Unused for this function
    :return: One-dimensional numpy array containing the fitness values of the chromosomes.
    """
    optimized_routes = np.zeros_like(chromosomes)
    progress_bar_range = enumerate(tqdm(chromosomes, desc=f"Generation: {generation}"))
    for index, chromosome in progress_bar_range:
        optimized_routes[index] = two_opt(flow_matrix, distance_matrix, chromosome)

    return bulk_basic_fitness_function(flow_matrix, distance_matrix, optimized_routes)

