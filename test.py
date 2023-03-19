import numpy as np
from Coalescence import Coalescence
from Fst import Fst


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


def check_produce_coalescence(mat: np.ndarray, guess):
    t_mat = Coalescence(mat)
    f_mat = t_mat.produce_fst()
    f = Fst(f_mat)
    new_t = f.produce_coalescence(x0=guess)
    return new_t


test_mat2 = np.array(([[1, 2, 3, 2], [2, 1, 2, 2], [3, 2, 1, 3], [2, 2, 3, 1]]))
g = [2, 3, 2, 2, 2, 3, 1, 1, 1, 1]
t = check_produce_coalescence(test_mat2, g)
print(t)

