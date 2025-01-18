# Solving [QAP](https://en.wikipedia.org/wiki/Quadratic_assignment_problem) with evolutionary algorithms
This repository documents the progress of developing a evolutionary algorithm to solve the QAP. The goal was to find a solution as close as possible to 44759294. This work is also accompanied by a written work, see Nothing to see here yet.

# Best Results

## 47092272
```Python
# Hyperparameters
POPULATION_SIZE: int = 100
NUMBER_OF_GENERATIONS: int = 3000
NUMBER_OF_FACILITIES: int = 256
MUTATION_PROB: float = 0.3
TOURNAMENT_SIZE: int = 25

# Functions used
basic_evolution_loop(
    bulk_basic_fitness_function, 
    roulette_wheel_selection, 
    partially_mapped_crossover, 
    swap_mutation, 
    TESTING
)
```

Best result:
```commandline
Best solution: 
[224   3 117 210 134  58 155 253 170  88 227 234  37  29  45  60 241 221
 146 189   1 212 157 136  54 201 160  20 107 165 163 178  84  94 236 207
  33 184 251  91 231 168  77  26 154  63 114 123  71 132 238  97 105  15
  59 151 127  80 115  11 180  73  18  28  34 182 125 246  31 159 248 161
  82 137  68  24 172 217 229 214 192  56 103 148  79   6 140 128 111 187
  52 204  93 113 130 118  72  12 245 232 183 181 243 175 133  41 186 206
  27  14 171 109  65 143   9 156 147  61  74 121 102 198  78 126  50 205
  92 223  42  70 188 185  16  51 200 226 228 199 138  36 191  17 194 197
  66 169 144  40 177 216 104 222 208  19 101 124  55  44  43  99 153  81
  95 225 141 247 139  90 129  57  30 209 254 215  25 131  10 190 122   7
 116  83  32 164 106 145 193 150   8   2  67 174  38 239 149 202  46 244
 112  47 249  53 195 120  75   5 176  86  62  48  96 108  13 196  87 152
 135 233 240 218 237 252 119 213 158  22 110 219  98 167 173 255 235 203
 250 211  35  85  21  23  89  49 230  39  69 179 220 242 100   0   4 142
  64  76 162 166] with fitness 47092272
```
