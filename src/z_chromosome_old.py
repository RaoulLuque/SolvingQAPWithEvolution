from typing import Callable

from src.config import NUMBER_OF_FACILITIES

import numpy as np

from src.serialization import write_chromosome_to_file


def random_with_fitness(flow_matrix: np.ndarray, distance_matrix: np.ndarray, fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], int]) -> tuple[np.ndarray, int]:
    """
    Create a random chromosome with a fitness value.
    """
    chromosome = np.random.permutation(np.eye(NUMBER_OF_FACILITIES))
    fitness = fitness_function(flow_matrix, distance_matrix, chromosome)

    return chromosome, fitness


def check_if_chromosome_is_valid(chromosome: np.ndarray) -> bool:
    """
    Check if the chromosome is valid.
    """
    unique, counts = np.unique(chromosome, return_counts=True, axis=0)
    duplicates = unique[counts > 1]
    # Check if there are any duplicate rows
    if len(duplicates) != 0:
        duplicate_indexes = [i for i, row in enumerate(chromosome) if
                             any(np.array_equal(row, dup) for dup in duplicates)]
        print(f"There are duplicate rows. The indexes are {duplicate_indexes}")

        write_chromosome_to_file("invalid_chromosome_duplicates", chromosome)
        return False
    for i in range(NUMBER_OF_FACILITIES):
        # Check if for all rows the sum of the row entries is 1
        if np.sum(chromosome[i]) != 1:
            print(f"Some rows don't sum to 1. Row: {i}")
            write_chromosome_to_file("invalid_chromosome_row_sum", chromosome)
            return False
    return True
