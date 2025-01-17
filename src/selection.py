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


def tournament_selection_two_tournament(population: np.ndarray, population_fitness: np.ndarray) -> np.ndarray:
    """
    A basic tournament selection algorithm.
    :param population: The population of chromosomes on which to perform selection
    :param population_fitness: The population's fitness values
    :param tournament_size: The size of the tournament
    :return: A numpy array of size len(population) containing the selected chromosomes
    """
    selected_chromosomes = np.empty_like(population)

    # selected_fighters_indexes = np.random.choice(len(population), 2 * len(population), replace=True)
    # selected_fighters = population[selected_fighters_indexes]
    # selected_fighters_fitness = population_fitness[selected_fighters_indexes]

    for i in range(len(population)):
        first_fighter, second_fighter = np.random.choice(len(population), 2, replace=False)
        random_number = np.random.rand()
        if population_fitness[first_fighter] > population_fitness[second_fighter]:
            selected_chromosomes[i] = population[first_fighter] if random_number < 0.9 else population[second_fighter]
        else:
            selected_chromosomes[i] = population[second_fighter] if random_number < 0.9 else population[first_fighter]

    return selected_chromosomes
