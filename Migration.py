import numpy as np
from Helper_funcs import comb as comb


class Migration:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize a migration matrix object
        :param matrix: input migration matrix
        """
        self.matrix = matrix

    def produce_coalescence(self) -> np.ndarray:
        """
        produces and returns the corresponding coalescence matrix
        :return: The corresponding coalescence matrix
        """
        A = self.produce_coefficient_matrix()
        b = self.produce_solution_vector()
        return np.linalg.solve(A, b)

    def calculate_first_coefficients(self, j: int, i: int, same_pop: int, lower_bound: int, upper_bound: int,
                                     p_list: list, counter: list) -> float:
        n = self.matrix.shape[0]
        if j == same_pop:
            return 1 + np.sum(self.matrix[i, :])
        if lower_bound <= j <= upper_bound:
            counter[0] += 1
            return -1 * self.matrix[i, i + counter[0] - 1]
        for p in p_list:
            if j == (i - p) + np.sum([n - k for k in range(p)]):
                return -1 * self.matrix[i, p]
        return 0

    def calculate_last_coefficients(self, j, cur_pop, other_pop) -> float:
        n = self.matrix.shape[0]
        if j == np.sum([n - k for k in range(other_pop)]) + (cur_pop - other_pop):
            return float(np.sum(self.matrix[[cur_pop, other_pop], :]))
        for p in range(n):
            for t in [other_pop, cur_pop]:
                if t == other_pop:
                    not_t = cur_pop
                else:
                    not_t = other_pop
                if p != not_t:
                    min_t_p = min(t, p)
                    max_t_p = max(t, p)
                    if j == np.sum([n - k for k in range(min_t_p)]) + max_t_p - min_t_p:
                        return -1 * self.matrix[not_t, p]
        return 0

    def produce_coefficient_matrix(self) -> np.ndarray:
        """
        produce and return the coefficient matrix used to calculate the T matrix(coalescence)
        :return: Coefficient matrix corresponding to object's migration matrix
        """
        n = self.matrix.shape[0]
        n_last_equations = comb(n, 2)
        n_first_equations = n
        mat_size = n_first_equations + n_last_equations
        coefficient_mat = np.zeros((mat_size, mat_size))
        for i in range(n_first_equations):
            same_population = int(np.sum([n - k for k in range(i)]))
            lower_bound = same_population + 1
            upper_bound = np.sum([n - k for k in range(i + 1)]) - 1
            smaller_ind_lst = [p for p in range(i)]
            counter = [1]
            for j in range(mat_size):
                coefficient_mat[i, j] = self.calculate_first_coefficients(j, i, same_population, lower_bound,
                                                                          upper_bound, smaller_ind_lst, counter)
        cur_population = 1
        other_population = 0
        for i in range(n_last_equations):
            if other_population == cur_population:
                other_population = 0
                cur_population += 1
            for j in range(mat_size):
                coefficient_mat[n + i, j] = self.calculate_last_coefficients(j, cur_population, other_population)
            other_population += 1

        return coefficient_mat

    def produce_solution_vector(self):
        n = self.matrix.shape[0]
        n_first = np.repeat(1, n)
        n_last = np.repeat(2, comb(n, 2))
        return np.hstack((n_first, n_last))
