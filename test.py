import numpy as np


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


t = np.array([[0.9, 1, 1], [1, 0.8, 0.8], [1, 0.8, 0.9]])
print(t)
print(check_constraint(t))
