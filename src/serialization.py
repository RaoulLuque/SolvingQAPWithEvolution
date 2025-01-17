import numpy as np


def write_chromosome_to_file(file_path: str, chromosome: np.ndarray, fitness: int | None = None):
    with open(f"{file_path}.txt", "w") as file:
        file.write(f"Fitness {fitness}\n")
        np.savetxt(f"{file_path}.txt", chromosome, fmt="%s")
