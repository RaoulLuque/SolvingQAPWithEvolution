import datetime
import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

from src.evolutionary_tools.chromosome import generate_random_chromosomes
from src.evolutionary_tools.greedy_optimizations import reset_cache, NUM_CACHE_HITS
from src.evolutionary_tools.mutation import apply_mutation_to_population, swap_mutation
from src.read_data import read_data
from src.config import POPULATION_SIZE, NUMBER_OF_GENERATIONS, TESTING, TESTING_SIZE, NUMBER_OF_FACILITIES, \
    MUTATION_PROB, TOURNAMENT_SIZE
from src.evolutionary_tools.fitness_function import bulk_basic_fitness_function, bulk_basic_fitness_function_baldwinian, \
    bulk_basic_fitness_function_lamarckian
from src.evolutionary_tools.recombine import recombine_chromosomes, order_crossing, partially_mapped_crossover, uniform_like_crossover_two
from src.evolutionary_tools.selection import roulette_wheel_selection, tournament_selection_two_tournament, \
    tournament_selection_two_tournament_bulk, tournament_selection_k_tournament_bulk, \
    tournament_selection_k_tournament_bulk_no_duplicates, tournament_selection_k_tournament_no_duplicates_unbiased

variants = ["standard", "baldwinian", "lamarckian"]
fitness_functions = ["bulk_basic"]
selection_functions = ["roulette_wheel", "tournament_two", "tournament_two_bulk", "tournament_k_bulk", "tournament_k_bulk_no_dups", "tournament_k_no_dups_unbiased"]
recombination_functions = ["order", "partially_mapped"]
mutation_functions = ["swap"]


def main():
    # Set config
    variant = "lamarckian"
    fitness_function_str = "bulk_basic"
    selection_function_str = "roulette_wheel"
    recombination_function_str = "partially_mapped"
    mutation_function_str = "swap"
    date = datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S')

    fitness_function, selection_function, recombination_function, mutation_function = translate_strings_to_functions(variant, fitness_function_str, selection_function_str, recombination_function_str, mutation_function_str)

    start_time = time.time()
    best_chromosome, best_fitness, best_fitness_each_generation, time_per_generation = basic_evolution_loop(fitness_function, selection_function, recombination_function, mutation_function, TESTING)
    end_time = time.time()

    total = end_time - start_time
    print(f"Total time: {total}")

    log_results(variant, fitness_function_str, selection_function_str, recombination_function_str, mutation_function_str, best_chromosome, best_fitness, total, date, time_per_generation, best_fitness_each_generation)
    plot_results(best_fitness_each_generation, variant, date)


def basic_evolution_loop(
    fitness_function: Callable[[ndarray, ndarray, ndarray, bool, int], tuple[ndarray, ndarray]],
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
    population, population_fitness = fitness_function(flow_matrix, distance_matrix, population, False)
    best_fitness_each_generation.append(np.min(population_fitness))

    for generation in range(NUMBER_OF_GENERATIONS):
        start_time = time.time()
        if generation % 10 == 0:
            print(f"Generation {generation + 1}")
            print(f"Best fitness: {np.min(population_fitness)}")

            number_of_unique_permutations = len(np.unique(population, axis=0))
            print(f"Number of unique permutations: {number_of_unique_permutations}")

            if len(time_per_generation) != 0:
                average_time_per_generation_per_individual = np.mean(time_per_generation) / POPULATION_SIZE
                print(f"Average time per generation per individual: {average_time_per_generation_per_individual}")

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

        # Evaluate the new population (and possibly apply Lamarckian evolution)
        if generation == NUMBER_OF_GENERATIONS - 1:
            population, population_fitness = fitness_function(flow_matrix, distance_matrix, population, True, generation + 1)
        else:
            population, population_fitness = fitness_function(flow_matrix, distance_matrix, population, False, generation + 1)

        # Force the fittest individual to survive by replacing worst in new population with best from last population
        worst_individual = np.argmax(population_fitness)
        population[worst_individual] = fittest_individual
        population_fitness[worst_individual] = fitness_of_fittest_individual

        # Add the fittest individual to the list
        best_fitness_each_generation.append(np.min(population_fitness))

        # Time generation
        end_time = time.time()
        time_per_generation.append(end_time - start_time)

        # Check if cache needs to be reset
        if generation % 125 == 0:
            reset_cache()

    print(f"Number of cache hits: {NUM_CACHE_HITS}")
    print(f"Best solution: {population[np.argmin(population_fitness)]} with fitness {np.min(population_fitness)}")

    return population[np.argmin(population_fitness)], np.min(population_fitness), best_fitness_each_generation, time_per_generation


def translate_strings_to_functions(variant: str, fitness_function_str: str, selection_function_str: str, recombination_function_str: str, mutation_function_str: str) -> tuple[Callable[[ndarray, ndarray, ndarray, bool, int], tuple[ndarray, ndarray]], Callable[[ndarray, ndarray, int], ndarray], Callable[[ndarray, ndarray], tuple[ndarray, ndarray]], Callable[[ndarray], ndarray]]:
    fitness_function, selection_function, recombination_function, mutation_function = None, None, None, None

    match fitness_function_str:
        case "bulk_basic":
            match variant:
                case "standard":
                    fitness_function = bulk_basic_fitness_function
                case "baldwinian":
                    fitness_function = bulk_basic_fitness_function_baldwinian
                case "lamarckian":
                    fitness_function = bulk_basic_fitness_function_lamarckian

    match selection_function_str:
        case "roulette_wheel":
            selection_function = roulette_wheel_selection
        case "tournament_two":
            selection_function = tournament_selection_two_tournament
        case "tournament_two_bulk":
            selection_function = tournament_selection_two_tournament_bulk
        case "tournament_k_bulk":
            selection_function = tournament_selection_k_tournament_bulk
        case "tournament_k_bulk_no_dups":
            selection_function = tournament_selection_k_tournament_bulk_no_duplicates
        case "tournament_k_no_dups_unbiased":
            selection_function = tournament_selection_k_tournament_no_duplicates_unbiased

    match recombination_function_str:
        case "order":
            recombination_function = order_crossing
        case "partially_mapped":
            recombination_function = partially_mapped_crossover

    match mutation_function_str:
        case "swap":
            mutation_function = swap_mutation

    if not all([fitness_function, selection_function, recombination_function, mutation_function]):
        raise ValueError("Invalid function string provided")

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
        best_fitness_each_generation: list[float],
):
    average_time_per_generation_per_individual = (np.mean(time_per_generation) / POPULATION_SIZE) * 1000

    # Cutoff total at 2 decimals
    total = round(total, 2)
    average_time_per_generation_per_individual = round(average_time_per_generation_per_individual, 3)

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
        file.write(f"Total time: {total} seconds\n")
        file.write(f"Average time per generation per individual: {average_time_per_generation_per_individual} milliseconds\n")
        file.write(f"Best chromosome: \n")
        best_chromosome = [int(gene) for gene in best_chromosome]
        file.write(f"{best_chromosome}\n")

    log_file_path = f"results/{date}_{variant}.log"
    with open(log_file_path, "w") as log_file:
        best_fitness_each_generation = [int(fitness) for fitness in best_fitness_each_generation]
        log_file.write(f"Best fitness each generation: \n {best_fitness_each_generation}\n")


def plot_results(best_fitness_each_generation: list[float], variant: str, date: str):
    plt.plot(best_fitness_each_generation)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.ylim(44759294, 53000000)
    plt.title("Fitness of the fittest individual in population over generations")
    plt.savefig(f"results/{date}_{variant}.png")
    plt.show()


if __name__ == "__main__":
    main()
