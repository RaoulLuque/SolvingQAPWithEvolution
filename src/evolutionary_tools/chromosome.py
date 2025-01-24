import numpy as np
from numpy import ndarray


def generate_random_chromosomes(number_of_chromosomes: int, number_of_facilities: int) -> ndarray:
    """
    Generate random chromosomes. Individual chromosomes are represented as a one-dimensional numpy array (permutation list)
    and generated using np.random.permutation.
    :param number_of_chromosomes: The number of chromosomes to generate
    :param number_of_facilities: The number of facilities (length of the permutation)
    :return: A two-dimensional numpy array representing the list of chromosomes
    """
    return np.array([np.random.permutation(np.arange(number_of_facilities)) for i in range(number_of_chromosomes)])
