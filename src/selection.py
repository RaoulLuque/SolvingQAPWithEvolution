import numpy as np


def roulette_wheel_selection(population: np.ndarray, population_fitness: np.ndarray) -> np.ndarray:
    """
    A basic roulette-wheel selection algorithm also known as fitness-proportionate selection.
    It selects chromosomes based on their fitness values, see https://en.wikipedia.org/wiki/Fitness_proportionate_selection
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :return: A numpy array of size len(population) containing the selected chromosomes (possibly containing duplicates)
    """
    total_fitness = np.sum(population_fitness)
    probabilities = population_fitness / total_fitness

    return population[np.random.choice(len(population), len(population), p=probabilities)]


def tournament_selection(population: np.ndarray, population_fitness: np.ndarray, tournament_size: int) -> np.ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    selected_chromosomes = np.empty_like(population)

    for i in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = population_fitness[tournament_indices]
        selected_chromosomes[i] = population[tournament_indices[np.argmin(tournament_fitness)]]

    return selected_chromosomes
