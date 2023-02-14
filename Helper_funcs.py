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
    added_eqs = 0
    eqs_lst = []
    nC2 = comb(n, 2)
    k = 0
    for i in range(nC2):
        for j in range(i + 1, n):
            eq = t[k] - (0.5 * (t[n + i] + t[n + j]) * ((1 + f[k]) / (1 - f[k])))
            eqs_lst.append(eq)
            # if added_eqs < n:
            #     eqs_lst.append(eq)
            #     added_eqs += 1
            k += 1
    # return np.repeat(t[0] - (0.5 * (t[1] + t[2]) * ((1 + f) / (1 - f))), 3)
    return eqs_lst
