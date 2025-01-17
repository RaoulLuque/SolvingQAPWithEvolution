from typing import Callable

import numpy as np

from src.read_data import read_data
from src.config import POPULATION_SIZE, NUMBER_OF_GENERATIONS
from src.population import generate_initial_population_with_fitness
from src.fitness_function import basic_fitness_function


def main():
    basic_evolution_loop(basic_fitness_function)


def basic_evolution_loop(fitness_function: Callable[[np.ndarray, np.ndarray, np.ndarray], int]):
    (flow_matrix, distance_matrix) = read_data()

    population, population_fitness = generate_initial_population_with_fitness(POPULATION_SIZE, fitness_function, flow_matrix, distance_matrix)

    best_individual_index = np.argmin(population_fitness)
    print(f"Best individual: {population[best_individual_index]}, Fitness: {population_fitness[best_individual_index]}")


if __name__ == "__main__":
    main()
