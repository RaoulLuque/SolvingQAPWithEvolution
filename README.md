# Solving [QAP](https://en.wikipedia.org/wiki/Quadratic_assignment_problem) with evolutionary algorithms
This repository documents the progress of developing a evolutionary algorithm to solve the QAP. The goal was to find a solution as close as possible to 44759294. This work is also accompanied by a written work, see Nothing to see here yet.

# Best Results

The best results are in descending order, that is, the best result is shown on top.

## [44792836](https://github.com/RaoulLuque/SolvingQAPWithEvolution/tree/0a10e2c95465b3e258575c765e1a0f9d1abe7dac)

```text
Functions used:
Variant: lamarckian
Fitness function: bulk_basic
Selection function: roulette_wheel
Recombination function: partially_mapped
Mutation function: swap

Hyperparameters:
Population size: 20
Number of generations: 1500
Number of facilities: 256
Mutation probability: 0.1
Tournament size: 10
Testing: False

Results:
Best fitness: 44792836
Total time: 36927.23 seconds
Average time per generation per individual: 1230.115 milliseconds
```

For the exact result see [best_result/best_result.txt](https://github.com/RaoulLuque/SolvingQAPWithEvolution/blob/0a10e2c95465b3e258575c765e1a0f9d1abe7dac/best_result/best_result.txt) from commit 0a10e2c.

## [44848506](https://github.com/RaoulLuque/SolvingQAPWithEvolution/tree/48a3a2939e831397f5e634c0fc9c8fdbe18e27cb)
```text
Functions used:
Variant: lamarckian
Fitness function: bulk_basic
Selection function: roulette_wheel
Recombination function: partially_mapped
Mutation function: swap

Hyperparameters:
Population size: 10
Number of generations: 25
Number of facilities: 256
Mutation probability: 0.1
Tournament size: 10
Testing: False

Results:
Best fitness: 44848506
Total time: 299.84 seconds
Average time per generation per individual: 1134.421 milliseconds
```

For the exact result see [best_result/best_result.txt](https://github.com/RaoulLuque/SolvingQAPWithEvolution/blob/48a3a2939e831397f5e634c0fc9c8fdbe18e27cb/best_result/best_result.txt) from commit 48a3a29.

## [46892802](https://github.com/RaoulLuque/SolvingQAPWithEvolution/tree/831c70e6a3e51d0eaf5ef58764b3edae08fe07fe)
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

For the exact result see [best_result/best_result.txt](https://github.com/RaoulLuque/SolvingQAPWithEvolution/blob/831c70e6a3e51d0eaf5ef58764b3edae08fe07fe/best_result/best_result.txt) from commit 831c70e.

## [47625886](https://github.com/RaoulLuque/SolvingQAPWithEvolution/tree/85fa67a4579f45fb683e58bdd0e1d28073cd3a78)
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

For the exact result see [best_result/best_result.txt](https://github.com/RaoulLuque/SolvingQAPWithEvolution/blob/85fa67a4579f45fb683e58bdd0e1d28073cd3a78/best_result/best_result.txt) from commit 85fa67a.
