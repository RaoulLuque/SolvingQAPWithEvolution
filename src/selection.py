import numpy as np
from numpy import ndarray


def roulette_wheel_selection(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
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


def tournament_selection_two_tournament(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    selected_chromosomes = np.empty_like(population)

    for i in range(len(population)):
        first_fighter, second_fighter = np.random.choice(len(population), 2, replace=False)
        random_number = np.random.rand()
        if population_fitness[first_fighter] > population_fitness[second_fighter]:
            selected_chromosomes[i] = population[first_fighter] if random_number < 0.9 else population[second_fighter]
        else:
            selected_chromosomes[i] = population[second_fighter] if random_number < 0.9 else population[first_fighter]

    return selected_chromosomes


def tournament_selection_two_tournament_bulk(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    selected_fighters_indexes = np.random.choice(len(population), 2 * len(population), replace=True)
    selected_fighters = population[selected_fighters_indexes]
    selected_fighters_fitness = population_fitness[selected_fighters_indexes]

    winners_mask = (selected_fighters_fitness[::2] > selected_fighters_fitness[1::2]) & (
                np.random.rand(len(population)) < 0.9)
    winners = np.where(winners_mask[:, np.newaxis], selected_fighters[::2], selected_fighters[1::2])

    return winners


def tournament_selection_k_tournament_bulk(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    selected_fighters_indexes = np.random.choice(len(population), tournament_size * len(population), replace=True)
    selected_fighters = population[selected_fighters_indexes]
    selected_fighters_fitness = population_fitness[selected_fighters_indexes]

    winners_mask = (selected_fighters_fitness[::tournament_size] > selected_fighters_fitness[1::tournament_size]) # & (np.random.rand(len(population)) < 0.9)
    winners = np.where(winners_mask[:, np.newaxis], selected_fighters[::tournament_size], selected_fighters[1::tournament_size])

    return winners


def tournament_selection_k_tournament_bulk_no_duplicates(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    population_size = len(population)

    # Create an array to hold the results
    selected_fighters_indexes = np.empty(population_size * tournament_size, dtype=int)

    # For each slot in the population, perform a random choice with no replacement
    for i in range(population_size):
        selected_fighters_indexes[i * tournament_size: (i + 1) * tournament_size] = np.random.choice(
            population_size, tournament_size, replace=False
        )

    selected_fighters = population[selected_fighters_indexes]
    selected_fighters_fitness = population_fitness[selected_fighters_indexes]

    winners_mask = (selected_fighters_fitness[::tournament_size] > selected_fighters_fitness[1::tournament_size]) # & (np.random.rand(len(population)) < 0.9)
    winners = np.where(winners_mask[:, np.newaxis], selected_fighters[::tournament_size], selected_fighters[1::tournament_size])

    return winners
