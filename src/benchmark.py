import datetime
import sys

from src.main import run_evolution_algorithm
from src import config

VARIANTS_FOR_BENCHMARK = ["baldwinian"]
PROBLEMS_TO_BENCHMARK = ["bur26a.dat", "chr18b.dat", "nug16a.dat", "tai60a.dat", "tai256c.dat"]


def main():
    # Set config
    fitness_function_str = "bulk_basic"
    selection_function_str = "roulette_wheel"
    recombination_function_str = "partially_mapped"
    mutation_function_str = "swap"

    for variant in VARIANTS_FOR_BENCHMARK:
        if variant == "standard":
            monkeypatch_config("POPULATION_SIZE", 100)
            monkeypatch_config("NUMBER_OF_GENERATIONS", 1000)
            monkeypatch_config("MUTATION_PROB", 0.3)
        if variant == "baldwinian" or variant == "lamarckian":
            monkeypatch_config("POPULATION_SIZE", 20)
            monkeypatch_config("NUMBER_OF_GENERATIONS", 250)
            monkeypatch_config("MUTATION_PROB", 0.1)
        for problem in PROBLEMS_TO_BENCHMARK:
            date = datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S')
            num = int(''.join(filter(str.isdigit, problem.split('.')[0])))
            monkeypatch_config("NUMBER_OF_FACILITIES", num)
            run_evolution_algorithm(variant, fitness_function_str, selection_function_str, recombination_function_str,
                                    mutation_function_str, date, problem, "results_benchmark")


def monkeypatch_config(key, value):
    # Patch the value in the config module
    setattr(config, key, value)

    # Update the attribute in all modules that imported it
    for module in sys.modules.values():
        if hasattr(module, key):
            setattr(module, key, value)


if __name__ == "__main__":
    main()
