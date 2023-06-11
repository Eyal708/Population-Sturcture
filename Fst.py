import numpy as np
from scipy.optimize import least_squares, minimize
from Helper_funcs import compute_coalescence, comb, constraint_generator


class Fst:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize Fst matrix object
        :param matrix: input Fst matrix
        """
        self.matrix = matrix
        self.shape = matrix.shape[0]

    def produce_coalescence(self, x0=None, constraint=False, bounds=(0, np.inf)) -> np.ndarray:
        """
        generates a possible corresponding coalescence times matrix and returns it.
        :param constraint: indicated whether the T matrix produced should be 'good'. default is False
        :param bounds: bounds for each variable T(i,j), default is (0, inf). bounds should be a tuple of two arrays,
        first is lower bounds for each variable, second is upper bounds for each variable. If bounds is a tuple of
        two scalars, the same bounds are applied for each variable.
        :param x0: initial guess for the variables, default is a random vector with bounds (0,2*n), where n is the
        size of the matrix (number of populations).
        :return: A possible corresponding Coalescence time matrix- T.
        """
        n, nc2 = self.shape, comb(self.shape, 2)
        if x0 is None:
            x0 = np.random.uniform(low=0, high=2 * n, size=(n + nc2,))
        T = np.zeros((n, n))
        f_values = self.matrix[np.triu_indices(n, 1)]
        # add constraints
        constraints = None
        if constraint:
            constraints = []
            row, col = 0, 1
            for i in range(nc2):
                constraint_1 = constraint_generator(i, nc2 + row)
                constraint_2 = constraint_generator(i, nc2 + col)
                col += 1
                constraints.append({"type": "ineq", "fun": constraint_1})
                constraints.append({"type": "ineq", "fun": constraint_2})
                if col == n:  # move to next row
                    row += 1
                    col = row + 1
        # constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - x[6]},
        #                {'type': 'ineq', 'fun': lambda x: x[0] - x[7]},
        #                {'type': 'ineq', 'fun': lambda x: x[1] - x[6]},
        #                {'type': 'ineq', 'fun': lambda x: x[1] - x[8]},
        #                {'type': 'ineq', 'fun': lambda x: x[2] - x[6]},
        #                {'type': 'ineq', 'fun': lambda x: x[2] - x[9]},
        #                {'type': 'ineq', 'fun': lambda x: x[3] - x[7]},
        #                {'type': 'ineq', 'fun': lambda x: x[3] - x[8]},
        #                {'type': 'ineq', 'fun': lambda x: x[4] - x[7]},
        #                {'type': 'ineq', 'fun': lambda x: x[4] - x[9]},
        #                {'type': 'ineq', 'fun': lambda x: x[5] - x[8]},
        #                {'type': 'ineq', 'fun': lambda x: x[5] - x[9]}]

        solution = minimize(compute_coalescence, x0=x0, args=(f_values, n), bounds=(n + nc2) * [(bounds[0], bounds[1])],
                            constraints=constraints)
        x = solution.x
        np.fill_diagonal(T, x[nc2:])
        row_indices, col_indices = np.triu_indices(n, 1)
        T[(row_indices, col_indices)] = x[0:nc2]
        T[(col_indices, row_indices)] = x[0:nc2]
        # T[np.triu_indices(n, 1)] = x[0:nc2]
        # T[np.tril_indices(n, -1)] = x[0:nc2]
        return T
