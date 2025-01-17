from typing import Callable

import numpy as np

from src.chromosome import generate_random_chromosomes, check_if_chromosome_is_valid
from src.mutation import apply_mutation_to_population, swap_mutation
from src.read_data import read_data
from src.config import POPULATION_SIZE, NUMBER_OF_GENERATIONS, TESTING, TESTING_SIZE, NUMBER_OF_FACILITIES, \
    MUTATION_PROB
from src.fitness_function import bulk_basic_fitness_function
from src.recombine import recombine_chromosomes, order_crossing, partially_mapped_crossover
from src.selection import roulette_wheel_selection, tournament_selection_two_tournament
from src.serialization import write_chromosome_to_file


def main():
    basic_evolution_loop(bulk_basic_fitness_function, tournament_selection_two_tournament, partially_mapped_crossover, swap_mutation, TESTING)


def basic_evolution_loop(
    fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    selection_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    recombination_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation_function: Callable[[np.ndarray], np.ndarray],
    testing: bool
):
    (flow_matrix, distance_matrix) = read_data()

    if testing:
        flow_matrix = flow_matrix[:TESTING_SIZE, :TESTING_SIZE]
        distance_matrix = distance_matrix[:TESTING_SIZE, :TESTING_SIZE]

    population = generate_random_chromosomes(POPULATION_SIZE, NUMBER_OF_FACILITIES)
    population_fitness = fitness_function(flow_matrix, distance_matrix, population)

    for generation in range(NUMBER_OF_GENERATIONS):
        print(f"Generation {generation + 1}")
        print(f"Best fitness: {np.min(population_fitness)}")

        number_of_unique_permutations = len(np.unique(population, axis=0))
        print(f"Number of unique permutations: {number_of_unique_permutations}")

        # Check if the fittest individual is a valid solution
        assert check_if_chromosome_is_valid(population[np.argmin(population_fitness)])

        # Take the fittest individual to secure a spot in the new generation
        index_of_fittest_individual = np.argmin(population_fitness)
        fittest_individual = population[index_of_fittest_individual]
        fitness_of_fittest_individual = population_fitness[index_of_fittest_individual]

        # Selection
        selected_chromosomes = selection_function(population, population_fitness)

        # Recombine
        population = recombine_chromosomes(selected_chromosomes, recombination_function)

        # Mutate
        population = apply_mutation_to_population(population, mutation_function, MUTATION_PROB)

        # Evaluate the new population
        population_fitness = fitness_function(flow_matrix, distance_matrix, population)

        # Force best individual into new population
        if not np.min(population_fitness) == population_fitness[0]:
            population[0] = fittest_individual
            population_fitness[0] = fitness_of_fittest_individual
        else:
            population[1] = fittest_individual
            population_fitness[1] = fitness_of_fittest_individual

    print(f"Best solution: {population[np.argmin(population_fitness)]} with fitness {np.min(population_fitness)}")
    write_chromosome_to_file("best_result", population[np.argmin(population_fitness)], np.min(population_fitness))


if __name__ == "__main__":
    main()
