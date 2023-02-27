import numpy as np
from Matrix_generator import generate_pseudo_random_fst_mat
from Coalescence import Coalescence
from Fst import Fst


def analyze_f_to_m(f: np.ndarray) -> list:
    """
    Check how many T matrices generated follow the within < inbetween rule, and the difference between the M matrices
    that come from T matrices that follow the rule, in comparison to M matrices that come from T matrices that don't
    follow the rule. also generates the cost function of the T->M transformation.
    :param f: Fst matrix on which to perform the analysis.
    :return: a list with relevant parameters.
    """
    good_mats = 0  # counts how many t matrices follow the constraint
    good_cost, bad_cost = 0, 0  # average cost for matrices that follow/ don't follow the constraint
    for i in range(1000):
        F = Fst(f)
        t = F.produce_coalescence()
        T = Coalescence(t)
        cost = T.produce_migration()[1].cost
        if check_constraint(t):
            good_mats += 1
            good_cost += cost
        else:
            bad_cost += cost
    return [good_mats / 1000 * 100, good_cost / 1000, bad_cost / 1000]


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


if __name__ == "__main__":
    # plot_transformations(size=3)
    for i in range(5):
        f = generate_pseudo_random_fst_mat(n=4)
        print(f, "\n")
        results = analyze_f_to_m(f)
        print(f"% of matrices that follow the constraint: {results[0]}\n")
        print(f"Average cost of the transformation T->M for T matrices that follow the constraint: {results[1]}\n")
        print(f"For T matrices that don't follow the constraint: {results[2]}\n")
