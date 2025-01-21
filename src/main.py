import datetime
import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

from src.chromosome import generate_random_chromosomes, check_if_chromosome_is_valid
from src.mutation import apply_mutation_to_population, swap_mutation
from src.read_data import read_data
from src.config import POPULATION_SIZE, NUMBER_OF_GENERATIONS, TESTING, TESTING_SIZE, NUMBER_OF_FACILITIES, \
    MUTATION_PROB, TOURNAMENT_SIZE
from src.fitness_function import bulk_basic_fitness_function
from src.recombine import recombine_chromosomes, order_crossing, partially_mapped_crossover, uniform_like_crossover_two
from src.selection import roulette_wheel_selection, tournament_selection_two_tournament, \
    tournament_selection_two_tournament_bulk, tournament_selection_k_tournament_bulk
from src.serialization import write_chromosome_to_file

variants = ["standard", "baldwinian", "lamarckian"]
fitness_functions = ["bulk_basic"]
selection_functions = ["roulette_wheel", "tournament_two", "tournament_two_bulk", "tournament_k_bulk", "tournament_k_bulk_no_dups"]
recombination_functions = ["order", "uniform_like", "partially_mapped"]
mutation_functions = ["swap"]


def main():
    # Set config
    variant = "standard"
    fitness_function = "bulk_basic"
    selection_function = "tournament_k_bulk_no_dups"
    recombination_function = "uniform_like"
    mutation_function = "swap"
    date = datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S')

    translate_strings_to_functions(fitness_function, selection_function, recombination_function, mutation_function)

    start_time = time.time()
    best_chromosome, best_fitness, best_fitness_each_generation, time_per_generation = basic_evolution_loop(variant, bulk_basic_fitness_function, roulette_wheel_selection, partially_mapped_crossover, swap_mutation, TESTING)
    end_time = time.time()

    total = end_time - start_time
    print(f"Total time: {total}")

    log_results(variant, fitness_function, selection_function, recombination_function, mutation_function, best_chromosome, best_fitness, total, date, time_per_generation)
    plot_results(best_fitness_each_generation, variant, date)


def basic_evolution_loop(
    variant: str,
    fitness_function: Callable[[ndarray, ndarray, ndarray], ndarray],
    selection_function: Callable[[ndarray, ndarray, int], ndarray],
    recombination_function: Callable[[ndarray, ndarray], tuple[ndarray, ndarray]],
    mutation_function: Callable[[ndarray], ndarray],
    testing: bool
) -> tuple[ndarray, float, list[float], list[float]]:
    (flow_matrix, distance_matrix) = read_data()

    best_fitness_each_generation = []
    time_per_generation = []

    if testing:
        flow_matrix = flow_matrix[:TESTING_SIZE, :TESTING_SIZE]
        distance_matrix = distance_matrix[:TESTING_SIZE, :TESTING_SIZE]

    population = generate_random_chromosomes(POPULATION_SIZE, NUMBER_OF_FACILITIES)
    population_fitness = fitness_function(flow_matrix, distance_matrix, population)
    best_fitness_each_generation.append(np.min(population_fitness))

    for generation in range(NUMBER_OF_GENERATIONS):
        start_time = time.time()
        if generation % 25 == 0:
            print(f"Generation {generation + 1}")
            print(f"Best fitness: {np.min(population_fitness)}")

            number_of_unique_permutations = len(np.unique(population, axis=0))
            print(f"Number of unique permutations: {number_of_unique_permutations}")

            if len(time_per_generation) != 0:
                average_time_per_generation_per_individual = np.mean(time_per_generation) / POPULATION_SIZE
                print(f"Average time per generation per individual: {average_time_per_generation_per_individual}")

        # Check if the fittest individual is a valid solution
        # assert check_if_chromosome_is_valid(population[np.argmin(population_fitness)])

        # Take the fittest individual to secure a spot in the new generation
        index_of_fittest_individual = np.argmin(population_fitness)
        fittest_individual = population[index_of_fittest_individual]
        fitness_of_fittest_individual = population_fitness[index_of_fittest_individual]

        # Selection
        selected_chromosomes = selection_function(population, population_fitness, TOURNAMENT_SIZE)

        # Check if parents for one child are equal in which case replace those
        alternative_selected_chromosomes = []
        alternative_chromosome_index = 0
        for i in range(0, len(selected_chromosomes), 2):
            while np.array_equal(selected_chromosomes[i], selected_chromosomes[i + 1]):
                if len(alternative_selected_chromosomes) == 0:
                    # This should only be encountered once
                    alternative_selected_chromosomes = selection_function(population, population_fitness, TOURNAMENT_SIZE)
                if alternative_chromosome_index != len(alternative_selected_chromosomes):
                    selected_chromosomes[i + 1] = alternative_selected_chromosomes[alternative_chromosome_index]
                    alternative_chromosome_index += 1
                else:
                    alternative_selected_chromosomes = selection_function(population, population_fitness, TOURNAMENT_SIZE)
                    alternative_chromosome_index = 0
                    selected_chromosomes[i + 1] = alternative_selected_chromosomes[alternative_chromosome_index]
                    alternative_chromosome_index += 1

        # Recombine
        population = recombine_chromosomes(selected_chromosomes, recombination_function)

        # Mutate
        population = apply_mutation_to_population(population, mutation_function, MUTATION_PROB)

        # Evaluate the new population
        population_fitness = fitness_function(flow_matrix, distance_matrix, population)

        # Force the fittest individual to survive by replacing worst in new population with best from last population
        worst_individual = np.argmax(population_fitness)
        population[worst_individual] = fittest_individual
        population_fitness[worst_individual] = fitness_of_fittest_individual

        # Add the fittest individual to the list
        best_fitness_each_generation.append(np.min(population_fitness))

        # Time generation
        end_time = time.time()
        time_per_generation.append(end_time - start_time)

    print(f"Best solution: {population[np.argmin(population_fitness)]} with fitness {np.min(population_fitness)}")

    return population[np.argmin(population_fitness)], np.min(population_fitness), best_fitness_each_generation, time_per_generation


def translate_strings_to_functions(fitness_function: str, selection_function: str, recombination_function: str, mutation_function: str) -> tuple[Callable[[ndarray, ndarray, ndarray], ndarray],Callable[[ndarray, ndarray, int], ndarray], Callable[[ndarray, ndarray], tuple[ndarray, ndarray]], Callable[[ndarray], ndarray]]:
    match fitness_function:
        case "bulk_basic":
            fitness_function = bulk_basic_fitness_function

    match selection_function:
        case "roulette_wheel":
            selection_function = roulette_wheel_selection
        case "tournament_two":
            selection_function = tournament_selection_two_tournament
        case "tournament_two_bulk":
            selection_function = tournament_selection_two_tournament_bulk
        case "tournament_k_bulk":
            selection_function = tournament_selection_k_tournament_bulk

    match recombination_function:
        case "order":
            recombination_function = order_crossing
        case "uniform_like":
            recombination_function = uniform_like_crossover_two
        case "partially_mapped":
            recombination_function = partially_mapped_crossover
    match mutation_function:
        case "swap":
            mutation_function = swap_mutation

    return fitness_function, selection_function, recombination_function, mutation_function


def log_results(
        variant: str,
        fitness_function: str,
        selection_function: str,
        recombination_function: str,
        mutation_function: str,
        best_chromosome: ndarray,
        best_fitness: float,
        total: float,
        date: str,
        time_per_generation: list[float],
):
    average_time_per_generation_per_individual = np.mean(time_per_generation) / POPULATION_SIZE

    file_path = f"results/{date}_{variant}.txt"
    with open(file_path, "w") as file:
        file.write("Functions used:\n")
        file.write(f"Variant: {variant}\n")
        file.write(f"Fitness function: {fitness_function}\n")
        file.write(f"Selection function: {selection_function}\n")
        file.write(f"Recombination function: {recombination_function}\n")
        file.write(f"Mutation function: {mutation_function}\n")
        file.write("\n")

        file.write("Hyperparameters:\n")
        file.write(f"Population size: {POPULATION_SIZE}\n")
        file.write(f"Number of generations: {NUMBER_OF_GENERATIONS}\n")
        file.write(f"Number of facilities: {NUMBER_OF_FACILITIES}\n")
        file.write(f"Mutation probability: {MUTATION_PROB}\n")
        file.write(f"Tournament size: {TOURNAMENT_SIZE}\n")
        file.write(f"Testing: {TESTING}\n")
        file.write("\n")

        file.write("Results:\n")
        file.write(f"Best fitness: {best_fitness}\n")
        file.write(f"Total time: {total}\n")
        file.write(f"Average time per generation per individual: {average_time_per_generation_per_individual}\n")
        file.write(f"Best chromosome: \n")
        np.savetxt(f"{file_path}", best_chromosome, fmt="%d", delimiter=",")


def plot_results(best_fitness_each_generation: list[float], variant: str, date: str):
    plt.plot(best_fitness_each_generation)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of the fittest individual in population over generations")
    plt.savefig(f"results/{date}_{variant}.png")
    plt.show()


if __name__ == "__main__":
    main()
