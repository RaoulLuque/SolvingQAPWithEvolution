import numpy as np
from numpy import ndarray


def read_data() -> tuple[ndarray, ndarray]:
    """
    Read the data from the data/tai256c.dat file and return the flow and distance matrices.
    :return: A tuple containing the flow and distance matrices
    """
    data = np.loadtxt('data/tai256c.dat', skiprows=1)
    flow_matrix = np.int32(data[:256])
    distance_matrix = np.int32(data[256:])
    return flow_matrix, distance_matrix
