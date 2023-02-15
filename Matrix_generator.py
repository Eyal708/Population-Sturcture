from Migration import Migration
from Coalescence import Coalescence
import numpy as np
import scipy as sp


def generate_random_migration_mat(n: int = 3, bounds=(0, 1), decimals=None) -> np.ndarray:
    """
    Generates and returns a random migration matrix that follows conservative migration constraint.
    :param n: size of the migration matrix, default is 3.
    :param bounds: lower and upper bounds on migration values. default is [0,1).
    :param decimals: number of decimals to round the matrix' values. if None no rounding will be done (default).
    :return: A random migration matrix with conservative migration.
    """
    is_legal = False
    mat, x = None, None
    while not is_legal:
        mat = np.zeros((n, n))
        random_mat = np.random.uniform(low=bounds[0], high=bounds[1], size=(n - 1, n))
        mat[0:n - 1, ] = random_mat
        np.fill_diagonal(mat, 0)
        # Create the coefficient matrix to find the missing values
        A = np.zeros((n, n - 1))
        np.fill_diagonal(A, 1)
        A[n - 1, :] = 1
        # create solution vector
        b = np.append(np.array([np.sum(mat[i, :]) - np.sum(mat[:, i]) for i in range(n - 1)]), np.sum(mat[:, n - 1]))
        x = sp.optimize.lsq_linear(A, b).x
        is_legal = not np.any(x < 0)
    mat[n - 1, 0:n - 1] = x
    if decimals is not None:
        return np.round(mat, decimals=decimals)
    return mat


def generate_pseudo_random_fst_mat(n: int = 3, bounds=(0, 1), decimals=None) -> np.ndarray:
    """
    generate a pseudo random Fst matrix from a random migration matrix that follows the conservative migration
    constraint.
    :param n: size of the matrix. default is 3.
    :param bounds: lower and upper bound on migration values. Default is [0,1).
    :param decimals:  number of decimals to round the matrix' values. if None no rounding will be done (default).
    :return: Pseudo random Fst matrix that originated from a random migration matrix.
    """
    m = generate_random_migration_mat(n, bounds)
    M = Migration(m)
    t = M.produce_coalescence()
    T = Coalescence(t)
    random_f = T.produce_fst()
    if decimals is not None:
        return np.round(random_f, decimals=decimals)
    return random_f

