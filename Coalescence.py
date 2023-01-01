import numpy as np


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

    def produce_migration(self) -> np.ndarray:
        """
        produces and returns the corresponding migration matrix
        :return: The corresponding migration matrix
        """
        pass

    def produce_variable_vector(self) -> np.ndarray:
        """
        produces and returns the variable vector x corresponding to the object's matrix
        :return: the variable vector x
        """

    def produce_coefficient_mat(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:

        """
         produces and returns the corresponding coefficient matrix, assuming xx(T) is invertible.
        :param x: variable vector constructed from coefficient
        :param b: solution vector
        :return: The corresponding coefficient matrix (A)
        """

        x = x.reshape(x.shape[0], 1)
        x_T = np.transpose(x)
        xx_T = x @ x_T
        xx_T_inv = np.linalg.inv(xx_T)
        return np.linalg.multi_dot([b, x_T, xx_T_inv])






























































