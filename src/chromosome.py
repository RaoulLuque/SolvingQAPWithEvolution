import numpy as np

from src.serialization import write_chromosome_to_file


def generate_random_chromosomes(number_of_chromosomes: int, number_of_facilities: int) -> ndarray:
    """
    Generate random chromosomes.
    :param number_of_chromosomes: The number of chromosomes to generate
    :param number_of_facilities: The number of facilities
    :return: A three-dimensional numpy array representing the chromosomes
    """
    return np.array([np.random.permutation(np.arange(number_of_facilities)) for i in range(number_of_chromosomes)])


def check_if_chromosome_is_valid(chromosome: ndarray) -> bool:
    """
    Check if the chromosome is valid.
    """
    chromosome = chromosome.copy()
    chromosome.sort()
    if np.array_equal(chromosome, np.arange(chromosome.shape[0])):
        return True
    print(f"Chromosome is invalid")
    write_chromosome_to_file("invalid_chromosome", chromosome)
    return False
