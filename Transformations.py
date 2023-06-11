import numpy as np
from Migration import Migration
from Coalescence import Coalescence
from Fst import Fst
from Helper_funcs import split_migration, reassemble_matrix


def find_fst(m: np.ndarray) -> np.ndarray:
    """
    Receives a migration matrix with one connected component(a squared, positive matrix with zeroes on the diagonal),
    and returns it's corresponding Fst matrix according to Wilkinson-Herbot's equations.
    :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
    :return: Corresponding Fst matrix according to Wilkinson-Herbot's equations. If there is no solution, an error will
    occur.
    """
    if m.shape[0] == 1:
        return np.array([[0]])
    M = Migration(m)
    t = M.produce_coalescence()
    T = Coalescence(t)
    return T.produce_fst()


def find_coalescence(m: np.ndarray) -> np.ndarray:
    """
       Receives a migration matrix with one connected component
       (a squared, positive matrix with zeroes on the diagonal), and returns it's corresponding Coalescent times
       (T) matrix according to Wilkinson-Herbot's equations.
       :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
       :return: Corresponding T matrix according to Wilkinson-Herbot's equations. If there is no solution,
       an error will occur.
       """
    if m.shape[0] == 1:
        return np.array([[1]])
    M = Migration(m)
    return M.produce_coalescence()


def m_to_f(m: np.ndarray) -> np.ndarray:
    """
    Receives a migration matrix(a squared, positive matrix with zeroes on the diagonal) with any number
    of connected components, and returns it's corresponding Fst matrix according to Wilkinson-Herbot's equations and
    Slatkin equations.
    :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
    :return: Corresponding Fst matrix according to Wilkinson-Herbot's equations. If there is no solution, an error will
    occur.
    """
    split = split_migration(m)
    sub_matrices, components = split[0], split[1]
    f_matrices = []
    for matrix in sub_matrices:
        f_matrices.append(find_fst(matrix))
    return reassemble_matrix(f_matrices, components, "fst")


def m_to_t(m: np.ndarray) -> np.ndarray:
    """
       Receives a migration matrix(a squared, positive matrix with zeroes on the diagonal) with any number
       of connected components, and returns it's corresponding Coalescent times (T) matrix according to
       Wilkinson-Herbot's equations.
       :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
       :return: Corresponding T matrix according to Wilkinson-Herbot's equations. If there is no solution,
       an error will occur.
       """
    split = split_migration(m)
    sub_matrices, components = split[0], split[1]
    t_matrices = []
    for matrix in sub_matrices:
        t_matrices.append(find_coalescence(matrix))
    return reassemble_matrix(t_matrices, components, "coalescence")
