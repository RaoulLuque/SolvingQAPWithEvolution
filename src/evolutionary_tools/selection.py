import numpy as np
from numpy import ndarray


def roulette_wheel_selection(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic roulette-wheel selection algorithm also known as fitness-proportionate selection.
    It selects chromosomes based on their fitness values, see https://en.wikipedia.org/wiki/Fitness_proportionate_selection
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: Unused for this function
    :return: A numpy array of size len(population) containing the selected chromosomes (possibly containing duplicates)
    """
    total_fitness = np.sum(population_fitness)
    probabilities = population_fitness / total_fitness

    return population[np.random.choice(len(population), len(population), p=probabilities)]


def tournament_selection_two_tournament(population: ndarray, population_fitness: ndarray, tournament_size: int) -> ndarray:
    """
    A basic biased tournament selection algorithm. The tournament size is fixed at 2. There might be fights of the same chromosome with itself.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament. Unused for this function
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
    A basic biased tournament selection algorithm optimized for vectorized computation.
    The tournament size is fixed at 2. There might be fights of the same chromosome with itself.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament. Unused for this function
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
    A basic biased tournament selection algorithm optimized for vectorized computation.
    The tournament size is set by tournament_size. There might be fights of the same chromosome with itself.
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
    A basic biased tournament selection algorithm optimized for vectorized computation.
    The tournament size is set by tournament_size. There will not be fights of the same chromsome with itself.
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


def tournament_selection_k_tournament_no_duplicates_unbiased(population: ndarray, population_fitness: ndarray, tournament_size: int, p: float = 0.5) -> ndarray:
    """
    A basic unbiased tournament selection algorithm optimized for vectorized computation.
    The tournament size is set by tournament_size. There will not be fights of the same chromsome with itself.
    The winner of a tournament is not selected as the fittest individual. Instead, the winner is the fittest individual with a probability of p.
    If it is not chosen, the second fittest individual is chosen with a probability of p and so on.
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

    winners = np.empty_like(population)
    for index in range(population_size):
        fighters_for_this_tournament = selected_fighters[index * tournament_size: (index + 1) * tournament_size]
        fighters_fitness_for_this_tournament = selected_fighters_fitness[index * tournament_size: (index + 1) * tournament_size]

        sorted_fighters_for_this_tournament = fighters_for_this_tournament[np.argsort(fighters_fitness_for_this_tournament)]

        current_index = 0
        while np.random.rand() < p and current_index < tournament_size - 1:
            current_index += 1
        winners[index] = sorted_fighters_for_this_tournament[current_index]

    return winners
