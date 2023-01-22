import numpy as np
from Helper_funcs import comb


class Coalescence:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize a coalescence times matrix object
        :param matrix: input Coalescence time matrix
        """
        self.matrix = matrix
        self.shape = matrix.shape[0]

    def produce_fst(self) -> np.ndarray:
        """
        produces and returns the corresponding Fst matrix
        :return: The corresponding Fst matrix
        """
        F_mat = np.zeros((self.shape, self.shape))
        for i in range(self.shape):
            for j in range(i + 1, self.shape):
                t_S = (self.matrix[i, i] + self.matrix[j, j]) / 2
                t_T = (self.matrix[i, j] + t_S) / 2
                if np.isinf(t_T):
                    F_i_j = 1
                else:
                    F_i_j = (t_T - t_S) / t_T
                F_mat[i, j], F_mat[j, i] = F_i_j, F_i_j
        return F_mat

    def produce_variable_vector(self) -> np.ndarray:
        """
        produces and returns the variable vector x corresponding to the object's matrix
        :return: the variable vector x
        """
        indices = np.triu_indices(self.shape)  # selects indices above(including) main diagonal
        x = self.matrix[indices]
        x = x.reshape(x.shape[0], 1)
        return x

    def produce_coefficient_mat(self) -> np.ndarray:
        """
        produces and returns the corresponding coefficient matrix, assuming xx(T) is invertible.
        :return: The corresponding coefficient matrix (A)
        """
        x = self.produce_variable_vector()
        b = self.produce_solution_vector()
        x_T = np.transpose(x)
        xx_T = x @ x_T
        if np.linalg.matrix_rank(xx_T) != xx_T.shape[0]:
            raise ValueError("Matrix xx_T is not invertible!")
        xx_T_inv = np.linalg.inv(xx_T)
        A = np.linalg.multi_dot([b, x_T, xx_T_inv])
        return A

    def produce_solution_vector(self):
        """
        produce the solution vector(b), according to Wilkinson-Herbot's equations
        :return: solution vector b
        """
        n = self.shape
        n_first = np.repeat(1, n)
        n_last = np.repeat(2, comb(n, 2))
        b = np.hstack((n_first, n_last))
        return b.reshape(b.shape[0], 1)

    def produce_migration(self) -> np.ndarray:
        """
        produce and return the migration matrix induced by the coefficient matrix A(which is induced by T).
        :return: Migration matrix corresponding to object's Coalescence matrix
        """
        A = self.produce_coefficient_mat()
        n = self.shape
        n_last_equations = comb(n, 2)
        n_first_equations = n
        mat_size = n
        M = np.zeros((mat_size, mat_size))
        for i in range(n_first_equations):
            same_population = int(np.sum([n - k for k in range(i)]))
            lower_bound = same_population + 1
            upper_bound = np.sum([n - k for k in range(i + 1)]) - 1
            smaller_ind_lst = [p for p in range(i)]
            counter = [1]
            for j in range(mat_size):
                self.fill_by_first_equations(j, i, lower_bound, upper_bound, smaller_ind_lst, counter, A, M)
        cur_population = 1
        other_population = 0
        for i in range(n_last_equations):
            if other_population == cur_population:
                other_population = 0
                cur_population += 1
            for j in range(mat_size):
                self.fill_by_last_equations(i, j, cur_population, other_population, A, M)
            other_population += 1

        return M

    def fill_by_first_equations(self, j: int, i: int, lower_bound: int, upper_bound: int,
                                p_list: list, counter: list, A: np.ndarray, M: np.ndarray) -> None:
        """
        fills migration matrix(M) according to coefficient matrix(A) according to the rules of first n equations
        :param j: column of coefficient matrx
        :param i: row of coefficient matrix
        :param lower_bound: j values in range [lower_bound, upper_bound] correspond to coefficient -M(i,i+counter)
        :param upper_bound: j values in range [lower_bound, upper_bound] correspond to coefficient -M(i,i+counter)
        :param p_list: all values that are smaller than i
        :param counter: counts the number of time j was in the interval [lower_bound,upper_bound]
        :param A: coefficient matrix(nC2 X nC2)
        :param M: migration matrix(nXn)
        :return: The coefficient in place [i,j], for i in [n-1]
        """
        n = self.matrix.shape[0]
        if lower_bound <= j <= upper_bound:
            counter[0] += 1
            # return -1 * self.matrix[i, i + counter[0] - 1]
            M[i, i + counter[0] - 1] = -1 * A[i, j]
        else:
            for p in p_list:
                if j == (i - p) + np.sum([n - k for k in range(p)]):
                    M[i, p] = -1 * A[i, j]

    def fill_by_last_equations(self, i: int, j: int, cur_pop: int, other_pop: int, A: np.ndarray,
                               M: np.ndarray) -> None:
        """
         fills migration matrix(M) according to coefficient matrix(A) according to the rules of first n equations
        :param j: the column in the migration matrix
        :param cur_pop: the index of the population that corresponds to the current value
        :param other_pop: the index of the other population that corresponds to the current value
        :param A: coefficient matrix(nC2 X nC2)
        :param M: migration matrix(n X n)

        """
        n = self.shape
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
                        M[not_t, p] = -1 * A[n + i, j]
