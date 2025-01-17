from typing import Callable

import numpy as np

from src.chromosome import check_if_chromosome_is_valid
from src.read_data import read_data
from src.config import POPULATION_SIZE, NUMBER_OF_GENERATIONS, TESTING, TESTING_SIZE
from src.population import generate_initial_population_with_fitness
from src.fitness_function import basic_fitness_function, bulk_basic_fitness_function
from src.recombine import two_point_crossover, recombine_chromosomes, partially_mapped_crossover, order_crossing
from src.selection import roulette_wheel_selection
from src.serialization import write_chromosome_to_file


def main():
    basic_evolution_loop(basic_fitness_function, roulette_wheel_selection, order_crossing, TESTING)


def basic_evolution_loop(
        fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], int],
        selection_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        recombination_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        testing: bool
):
    (flow_matrix, distance_matrix) = read_data()

    if testing:
        flow_matrix = flow_matrix[:TESTING_SIZE, :TESTING_SIZE]
        distance_matrix = distance_matrix[:TESTING_SIZE, :TESTING_SIZE]

    population, population_fitness = generate_initial_population_with_fitness(POPULATION_SIZE, fitness_function, flow_matrix, distance_matrix)

    for generation in range(NUMBER_OF_GENERATIONS):
        print(f"Generation {generation + 1}")
        print(f"Best fitness: {np.min(population_fitness)}")

        # Check if the fittest individual is a valid solution
        assert check_if_chromosome_is_valid(population[np.argmin(population_fitness)])

        # Selection
        selected_chromosomes = selection_function(population, population_fitness)

        # Recombine
        population = recombine_chromosomes(selected_chromosomes, recombination_function)

        # Evaluate the new population
        population_fitness = bulk_basic_fitness_function(flow_matrix, distance_matrix, population)

    print(f"Best solution: {population[np.argmin(population_fitness)]} with fitness {np.min(population_fitness)}")
    write_chromosome_to_file("best_result", population[np.argmin(population_fitness)], np.min(population_fitness))


if __name__ == "__main__":
    main()
