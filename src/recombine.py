from typing import Callable

import numpy as np

from src.config import NUMBER_OF_FACILITIES


def recombine_chromosomes(selected_chromosomes: np.ndarray, recombination_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    new_population = np.empty_like(selected_chromosomes)
    for i in range(0, len(selected_chromosomes), 2):
        child_one, child_two = recombination_function(selected_chromosomes[i], selected_chromosomes[i + 1])
        new_population[i] = child_one
        new_population[i + 1] = child_two
    return new_population


def two_point_crossover(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform two-point crossover on the two parents.
    """
    # Get the length of the chromosome
    chromosome_length = NUMBER_OF_FACILITIES

    # Generate two random indices
    crossover_points = np.random.choice(chromosome_length, 2, replace=False)
    crossover_points.sort()

    # Perform the crossover
    child_one = np.copy(parent_one)
    child_two = np.copy(parent_two)

    child_one[crossover_points[0]:crossover_points[1]] = parent_two[crossover_points[0]:crossover_points[1]]
    child_two[crossover_points[0]:crossover_points[1]] = parent_one[crossover_points[0]:crossover_points[1]]

    repair(child_one)
    repair(child_two)

    return child_one, child_two


def repair(child: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(child, return_counts=True)
    duplicates = unique[counts > 1]
    missing = set(range(NUMBER_OF_FACILITIES)) - set(child)
    for duplicate in duplicates:
        duplicate_indices = np.where(child == duplicate)[0]
        for i in range(1, len(duplicate_indices)):
            child[duplicate_indices[i]] = missing.pop()
    return child
