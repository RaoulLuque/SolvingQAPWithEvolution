# Hyperparameters
POPULATION_SIZE: int = 100
NUMBER_OF_GENERATIONS: int = 1000
NUMBER_OF_FACILITIES: int = 256
MUTATION_PROB: float = 0.2

# Testing
TESTING: bool = False
TESTING_SIZE: int = 6

if TESTING:
    POPULATION_SIZE = TESTING_SIZE
    NUMBER_OF_GENERATIONS = TESTING_SIZE
    NUMBER_OF_FACILITIES = TESTING_SIZE
