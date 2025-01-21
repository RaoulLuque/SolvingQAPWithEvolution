import numpy as np
from numpy import ndarray


def write_chromosome_to_file(file_path: str, chromosome: ndarray, fitness: int | None = None):
    with open(f"{file_path}.txt", "w") as file:
        file.write(f"Fitness {fitness}\n")
        np.savetxt(f"{file_path}.txt", chromosome, fmt="%d", delimiter=",")
