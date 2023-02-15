import numpy as np
from scipy.optimize import least_squares
from Helper_funcs import compute_coalescence, comb


class Fst:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize Fst matrix object
        :param matrix: input Fst matrix
        """
        self.matrix = matrix
        self.shape = matrix.shape[0]

    def produce_coalescence(self, x0=None, bounds=(0, np.inf)) -> np.ndarray:
        """
        generates a possible  corresponding coalescence times matrix and returns it.
        :param bounds: bounds for each variable T(i,j), default is (0, inf). bounds should be a tuple of two arrays,
        first is lower bounds for each variable, second is upper bounds for each variable. If bounds is a tuple of
        two scalars, the same bounds are applied for each variable.
        :param x0: initial guess for the variables, default is an array of ones.
        :return: A possible corresponding Coalescence time matrix- T.
        """
        n, nc2 = self.shape, comb(self.shape, 2)
        if x0 is None:
            x0 = np.repeat(1, n + nc2)
        T = np.zeros((n, n))
        f_values = self.matrix[np.triu_indices(n, 1)]
        x = least_squares(compute_coalescence, x0=x0, args=(f_values, n), bounds=(bounds[0], bounds[1])).x
        np.fill_diagonal(T, x[nc2:])
        T[np.triu_indices(n, 1)] = x[0:nc2]
        T[np.tril_indices(n, -1)] = x[0:nc2]
        return T
