# Solving [QAP](https://en.wikipedia.org/wiki/Quadratic_assignment_problem) with evolutionary algorithms
This repository documents the progress of developing a evolutionary algorithm to solve the QAP. The goal was to find a solution as close as possible to 44759294. This work is also accompanied by a written work, see Nothing to see here yet.

# Best Results

The best results are in descending order, that is, the best result is shown on top.

## 46892802
```text
Functions used:
Variant: standard
Fitness function: bulk_basic
Selection function: roulette_wheel
Recombination function: partially_mapped
Mutation function: swap

Hyperparameters:
Population size: 200
Number of generations: 3000
Number of facilities: 256
Mutation probability: 0.1
Tournament size: 10
Testing: False

Results:
Best fitness: 46892802
Total time: 461.131915807724
Average time per generation per individual: 0.0007684138774871827
```

For the exact result see [best_result/best_result.txt](best_result/best_result.txt).

## 47625886
```text
Functions used:
Variant: standard
Fitness function: bulk_basic
Selection function: roulette_wheel
Recombination function: partially_mapped
Mutation function: swap

Hyperparameters:
Population size: 200
Number of generations: 3000
Number of facilities: 256
Mutation probability: 0.3
Tournament size: 25
Testing: False

Results:
Best fitness: 47625886
Total time: 489.8303732872009
```

For the exact result see [best_result/best_result.txt](best_result/best_result.txt).
