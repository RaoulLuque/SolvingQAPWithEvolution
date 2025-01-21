# Hyperparameters
POPULATION_SIZE: int = 200
NUMBER_OF_GENERATIONS: int = 100
NUMBER_OF_FACILITIES: int = 256
MUTATION_PROB: float = 0.3
TOURNAMENT_SIZE: int = 25

# Testing
TESTING: bool = False
TESTING_SIZE: int = 6

if TESTING:
    POPULATION_SIZE = TESTING_SIZE
    NUMBER_OF_GENERATIONS = TESTING_SIZE
    NUMBER_OF_FACILITIES = TESTING_SIZE
