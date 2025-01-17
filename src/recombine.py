from typing import Callable

import numpy as np


def recombine_chromosomes(selected_chromosomes: np.ndarray, recombination_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    new_population = np.empty_like(selected_chromosomes)
    for i in range(0, len(selected_chromosomes), 2):
        child_one, child_two = recombination_function(selected_chromosomes[i], selected_chromosomes[i + 1])
        new_population[i] = child_one
        new_population[i + 1] = child_two
    return new_population


def create_children_by_copying_crossover_part_from_parents(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Get the length of the chromosome
    chromosome_length = parent_one.shape[0]

    # Generate two random indices
    crossover_points = np.random.choice(chromosome_length, 2, replace=False)
    crossover_points.sort()

    child_one = np.copy(parent_one)
    child_two = np.copy(parent_two)

    child_one[crossover_points[0]:crossover_points[1]] = parent_one[crossover_points[0]:crossover_points[1]]
    child_two[crossover_points[0]:crossover_points[1]] = parent_two[crossover_points[0]:crossover_points[1]]

    return child_one, child_two, crossover_points


def order_crossing(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform order crossing on the two parents.
    """
    chromosome_length = parent_one.shape[0]

    child_one, child_two, crossover_points = create_children_by_copying_crossover_part_from_parents(parent_one, parent_two)

    # Perform the crossover
    child_one_parent_index = crossover_points[1]
    child_two_parent_index = crossover_points[1]

    child_index = crossover_points[1]
    # Loop once for every missing field in the childs
    for _ in range(chromosome_length - (crossover_points[1] - crossover_points[0])):
        while parent_two[child_one_parent_index] in child_one[crossover_points[0]:crossover_points[1]]:
            child_one_parent_index = (child_one_parent_index + 1) % chromosome_length
        child_one[child_index] = parent_two[child_one_parent_index]
        child_one_parent_index = (child_one_parent_index + 1) % chromosome_length

        while parent_one[child_two_parent_index] in child_two[crossover_points[0]:crossover_points[1]]:
            child_two_parent_index = (child_two_parent_index + 1) % chromosome_length
        child_two[child_index] = parent_one[child_two_parent_index]
        child_two_parent_index = (child_two_parent_index + 1) % chromosome_length

        child_index = (child_index + 1) % chromosome_length
    return child_one, child_two
