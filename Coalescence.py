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
