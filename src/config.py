# Hyperparameters
POPULATION_SIZE: int = 10
NUMBER_OF_GENERATIONS: int = 25
NUMBER_OF_FACILITIES: int = 256
MUTATION_PROB: float = 0.1
TOURNAMENT_SIZE: int = 10
NUMBER_OF_ITERATIONS_FOR_OPT: int = 1

# Testing
TESTING: bool = False
TESTING_SIZE: int = 6

if TESTING:
    POPULATION_SIZE = TESTING_SIZE
    NUMBER_OF_GENERATIONS = TESTING_SIZE
    NUMBER_OF_FACILITIES = TESTING_SIZE
