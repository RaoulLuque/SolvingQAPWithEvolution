import numpy as np


def read_data():
    data = np.loadtxt('data/tai256c.dat', skiprows=1)
    flow_matrix = np.int32(data[:256])
    distance_matrix = np.int32(data[256:])
    return flow_matrix, distance_matrix
