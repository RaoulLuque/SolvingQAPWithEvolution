import numpy as np


def find_row_if_exists(matrix, row) -> int | None:
    dtype = np.dtype((np.void, matrix.strides[0]))
    matches = np.where(matrix.astype('float64').view(dtype) == row.astype('float64').view(dtype))[0]

    return matches[0] if matches.size > 0 else None

if __name__ == "__main__":
    identity_matrix = np.eye(3)
    print(identity_matrix)

    row = np.array([0, 0, 1]).astype('float64')
    # print(any(np.equal(identity_matrix, row).all(1)))

    row_two = np.array([0, 2, 0])
    print(find_row_if_exists(identity_matrix, row))
    print(find_row_if_exists(identity_matrix, row_two))

