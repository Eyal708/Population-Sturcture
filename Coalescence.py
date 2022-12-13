import numpy as np


class Coalescence:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize a coalescence times matrix object
        :param matrix: input Coalescence time matrix
        """
        self.matrix = matrix

    def produce_fst(self) -> np.ndarray:
        """
        produces and returns the corresponding Fst matrix
        :return: The corresponding Fst matrix
        """
        pass

    def produce_migration(self) -> np.ndarray:
        """
        produces and returns the corresponding migration matrix
        :return: The corresponding migration matrix
        """
        pass
