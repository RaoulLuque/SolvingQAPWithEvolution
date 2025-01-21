import random
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

    child_one = -1 * np.ones_like(parent_one)
    child_two = -1 * np.ones_like(parent_two)

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


def partially_mapped_crossover(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform partially mapped crossover on the two parents.
    """
    chromosome_length = parent_one.shape[0]

    child_one, child_two, crossover_points = create_children_by_copying_crossover_part_from_parents(parent_one, parent_two)

    parent_one_list = list(parent_one)
    parent_two_list = list(parent_two)

    # Perform the crossover
    for i in range(crossover_points[0], crossover_points[1]):
        if parent_two[i] not in child_one[crossover_points[0]:crossover_points[1]]:
            index = parent_two_list.index(child_one[i])
            while crossover_points[0] <= index < crossover_points[1]:
                index = parent_two_list.index(child_one[index])
            child_one[index] = parent_two[i]

        if parent_one[i] not in child_two[crossover_points[0]:crossover_points[1]]:
            index = parent_one_list.index(child_two[i])
            while crossover_points[0] <= index < crossover_points[1]:
                index = parent_one_list.index(child_two[index])
            child_two[index] = parent_one[i]

    mask = child_one[:] == -1
    child_one[mask] = parent_two[mask]

    mask = child_two[:] == -1
    child_two[mask] = parent_one[mask]

    return child_one, child_two


def uniform_like_crossover_two(parent_one: np.ndarray, parent_two: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs Uniform-Like Crossover (ULX) on two parent chromosomes to generate two children.
    """
    child_one = uniform_like_crossover(parent_one, parent_two)
    child_two = uniform_like_crossover(parent_one, parent_two)

    return child_one, child_two


def uniform_like_crossover(parent_one: np.ndarray, parent_two: np.ndarray) -> np.ndarray:
    """
    Generates a child chromosome from two parents using the ULX algorithm.
    """
    size = len(parent_one)
    unassigned_indexes = []
    child = np.full(size, -1, dtype=int)  # Initialize child with -1

    # First pass: Inherit directly when parents agree or make a valid random choice
    for i in range(size):
        if parent_one[i] == parent_two[i]:
            child[i] = parent_one[i]
        elif parent_one[i] not in child and parent_two[i] not in child:
            child[i] = random.choice([parent_one[i], parent_two[i]])
        else:
            unassigned_indexes.append(i)

    # Collect unassigned chromosomes
    assigned_chromosomes = set(child[child != -1])
    unassigned_chromosomes = [chrom for chrom in range(size) if chrom not in assigned_chromosomes]

    # Shuffle unassigned elements for random assignment
    random.shuffle(unassigned_indexes)
    random.shuffle(unassigned_chromosomes)

    # Assign remaining chromosomes
    for index, chromosome in zip(unassigned_indexes, unassigned_chromosomes):
        child[index] = chromosome

    return child
