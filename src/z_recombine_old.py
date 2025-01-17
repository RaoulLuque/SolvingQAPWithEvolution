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


def order_crossing(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform order crossing on the two parents.
    """
    # Get the length of the chromosome
    chromosome_length = parent_one.shape[0]

    # Generate two random indices
    crossover_points = np.random.choice(chromosome_length, 2, replace=False)
    crossover_points.sort()

    # Perform the crossover
    child_one = np.copy(parent_one)
    child_two = np.copy(parent_two)

    child_one[crossover_points[0]:crossover_points[1]] = parent_one[crossover_points[0]:crossover_points[1]]
    child_two[crossover_points[0]:crossover_points[1]] = parent_two[crossover_points[0]:crossover_points[1]]

    child_one_index = crossover_points[1]
    child_two_index = crossover_points[1]

    for i in range(chromosome_length):
        if not any(np.equal(child_one[crossover_points[0]:crossover_points[1]], parent_two[i]).all(1)):
        # if parent_two[i] not in child_one[crossover_points[0]:crossover_points[1]]:
            child_one[child_one_index] = parent_two[i]
            child_one_index = (child_one_index + 1) % chromosome_length

        if not any(np.equal(child_two[crossover_points[0]:crossover_points[1]], parent_one[i]).all(1)):
        # if parent_one[i] not in child_two[crossover_points[0]:crossover_points[1]]:
            child_two[child_two_index] = parent_one[i]
            child_two_index = (child_two_index + 1) % chromosome_length
    return child_one, child_two


def partially_mapped_crossover(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform partially mapped crossover on the two parents.
    """
    # Get the length of the chromosome
    chromosome_length = NUMBER_OF_FACILITIES

    # Generate two random indices
    crossover_points = np.random.choice(chromosome_length, 2, replace=False)
    crossover_points.sort()

    # Perform the crossover
    child_one = 2 * np.ones_like(parent_one)
    child_two = 2 * np.ones_like(parent_two)

    child_one[crossover_points[0]:crossover_points[1]] = parent_one[crossover_points[0]:crossover_points[1]]
    child_two[crossover_points[0]:crossover_points[1]] = parent_two[crossover_points[0]:crossover_points[1]]

    perform_crossover_alt(parent_two, child_one, crossover_points)
    perform_crossover_alt(parent_one, child_two, crossover_points)

    return child_one, child_two


def perform_crossover(other_parent: np.ndarray, child: np.ndarray, crossover_points):
    for i in range(crossover_points[0], crossover_points[1]):
        if True not in np.all(child == other_parent[i], axis=1):
            index = np.where((child[i] == other_parent).all(axis=1))[0][0]
            while child[index][0] != 2:
                index = np.where((child[i] == other_parent).all(axis=1))[0][0]
            child[index] = other_parent[i]

    mask = child[:, 0] == 2
    child[mask] = other_parent[mask]


def perform_crossover_alt(other_parent: np.ndarray, child: np.ndarray, crossover_points):
    for i in range(crossover_points[0], crossover_points[1]):
        row_index = find_row_if_exists(child, other_parent[i])
        if row_index is not None:
            row_index = find_row_if_exists(other_parent, child[i])
            while child[row_index][0] != 2:
                row_index = find_row_if_exists(other_parent, child[i])
            child[row_index] = other_parent[i]
    mask = child[:, 0] == 2
    child[mask] = other_parent[mask]


def find_row_if_exists(matrix, row) -> int | None:
    dtype = np.dtype((np.void, matrix.strides[0]))
    matches = np.where(matrix.astype('float64').view(dtype) == row.astype('float64').view(dtype))[0]

    return matches[0] if matches.size > 0 else None


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
    unique, counts = np.unique(child, return_counts=True, axis=0)
    duplicates = unique[counts > 1]

    indices = np.argmax(child == 1, axis=1).tolist()
    missing = set(range(NUMBER_OF_FACILITIES)) - set(indices)
    for duplicate in duplicates:
        duplicate_indices = np.where((child == duplicate).all(axis=1))[0]
        for i in range(1, len(duplicate_indices)):
            child[duplicate_indices[i]] = create_array_with_one_at_index(child.shape[1], missing.pop())
    return child


def create_array_with_one_at_index(size: int, index: int) -> np.ndarray:
    array = np.zeros(size, dtype=int)
    array[index] = 1
    return array

