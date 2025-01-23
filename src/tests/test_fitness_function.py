import numpy as np

from src.evolutionary_tools.chromosome import generate_random_chromosomes
from src.evolutionary_tools.fitness_function import bulk_basic_fitness_function, basic_fitness_function
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


def test_basic_fitness_function_value_of_best():
    flow_matrix, distance_matrix = read_data()

    chromosome = [49, 111, 251, 87, 188, 46, 203, 229, 130, 216, 112, 64, 55, 7, 169, 158, 63, 246, 20, 207, 67, 37, 143, 32, 72, 77, 34, 124, 244, 238, 60, 43, 218, 171, 178, 139, 52, 57, 190, 109, 12, 255, 97, 69, 104, 115, 9, 106, 14, 210, 136, 236, 173, 3, 40, 74, 102, 22, 166, 199, 248, 141, 214, 161, 121, 186, 94, 242, 147, 29, 154, 184, 117, 176, 17, 144, 0, 221, 91, 208, 164, 151, 181, 84, 233, 132, 134, 225, 195, 212, 26, 82, 254, 83, 172, 116, 157, 2, 128, 133, 18, 28, 73, 44, 135, 187, 140, 90, 232, 226, 56, 110, 146, 211, 185, 167, 142, 66, 162, 53, 252, 168, 219, 145, 65, 118, 86, 101, 180, 127, 107, 220, 152, 41, 192, 96, 62, 228, 247, 71, 23, 105, 155, 138, 33, 75, 125, 222, 99, 19, 198, 6, 174, 163, 191, 196, 230, 38, 234, 122, 159, 1, 13, 165, 48, 31, 25, 241, 24, 205, 45, 15, 113, 30, 119, 253, 182, 249, 88, 120, 123, 85, 61, 16, 194, 103, 160, 202, 47, 156, 79, 76, 21, 95, 170, 177, 240, 213, 189, 80, 215, 200, 92, 223, 175, 131, 239, 201, 148, 108, 204, 68, 179, 235, 42, 217, 58, 183, 209, 5, 10, 35, 250, 126, 98, 227, 206, 231, 237, 245, 114, 78, 50, 59, 70, 4, 93, 54, 197, 89, 137, 150, 39, 8, 11, 193, 27, 81, 243, 224, 100, 149, 51, 36, 153, 129]
    chromosome = np.array(chromosome)

    expected = 44792836
    assert basic_fitness_function(flow_matrix, distance_matrix, chromosome) == expected

