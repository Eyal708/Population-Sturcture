import numpy as np


class Fst:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize Fst matrix object
        :param matrix: input Fst matrix
        """
        self.matrix = matrix

    def produce_coalescence(self) -> np.ndarray:
        """
        generated the corresponding Coalescence times matrix and returns it
        :return: The corresponding Coalescence time matrix
        """
        pass
