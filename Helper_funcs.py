import math
import numpy as np


def comb(n: int, k: int) -> int:
    """
    calculate and return n Choose k
    :param n: number of objects
    :param k: number of selected objects
    :return: n Choose k
    """
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def compute_coalescence(t: np.ndarray, f: np.ndarray, n: int) -> list:
    """
    returns the equations that describe the connection between coalescent times and Fst
    of all populations 1,2...n (Slatkin). These are the equations to minimize in order to find possible T matrices.
    :param t: an array representing [T(1,2),T(1,3)...,T(1,n),T(2,3)...,T(2,n),...T(1,1),T(2,2),...,T(n,n)], which are
    the variables to solve. Size of the array(number of unknowns) in nC2 + n.
    :param f: array of Fst values [F(1,2),F(1,3),,,,F(1,n),F(2,3),...F(2,n),...F(n-1,n)]. Array size is nC2.
    :param n: number of populations.
    :return: A list of all the equations that describe the connection between coalescent times and Fst
    of all populations 1,2...n (Slatkin).
    """
    # added_eqs = 0
    eqs_lst = []
    nC2 = comb(n, 2)
    k = 0
    for i in range(nC2):
        for j in range(i + 1, n):
            eq = t[k] - (0.5 * (t[nC2 + i] + t[nC2 + j]) * ((1 + f[k]) / (1 - f[k])))
            eqs_lst.append(eq)
            # if added_eqs < n:
            #     eqs_lst.append(eq)
            #     added_eqs += 1
            k += 1
    # return np.repeat(t[0] - (0.5 * (t[1] + t[2]) * ((1 + f) / (1 - f))), 3)
    return eqs_lst


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


def matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the distance between two matrices. Matrices must be of the same shape.
    :param a: first matrix.
    :param b: second matrix.
    :return: The distance between a and b.
    """
    n = a.shape[0]
    c = np.abs(a - b)
    return float(np.sum(c)) / (n ** 2 - n)


def diameter(mats: list) -> float:
    """
    Calculates the diameter for a given set of matrices.
    :param mats: list containing a set of matrices of the same shape.
    :return: The diameter (maximum pair-wise distance) of the set of matrices 'mats'.
    """
    max_diam = 0
    for i in range(len(mats)):
        for j in range(i):
            max_diam = max(max_diam, matrix_distance(mats[i], mats[j]))
    return max_diam
