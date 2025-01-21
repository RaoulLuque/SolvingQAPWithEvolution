import numpy as np

from src.chromosome import generate_random_chromosomes
from src.fitness_function import bulk_basic_fitness_function, basic_fitness_function
from src.read_data import read_data


def test_bulk_basic_fitness_function():
    flow_matrix, distance_matrix = read_data()
    population = generate_random_chromosomes(1, 256)

    fitness_values = bulk_basic_fitness_function(flow_matrix, distance_matrix, population)[1]
    # assert len(fitness_values) == 100

    expected = np.empty_like(fitness_values)
    for chromosome_index in range(population.shape[0]):
        expected[chromosome_index] = basic_fitness_function(flow_matrix, distance_matrix, population[chromosome_index])
    print(expected)
    assert np.array_equal(fitness_values, expected)


if __name__ == "__main__":
    test_bulk_basic_fitness_function()

