import numpy as np
import matplotlib.pyplot as plt
from Matrix_generator import generate_pseudo_random_fst_mat
from Coalescence import Coalescence
from Fst import Fst


def analyze_t_matrices(f: np.ndarray, n=1000) -> list:
    """
    Check how many T matrices generated follow the within < inbetween rule, and the difference between the M matrices
    that come from T matrices that follow the rule, in comparison to M matrices that come from T matrices that don't
    follow the rule. also generates the cost function of the T->M transformation.
    :param f: Fst matrix on which to perform the analysis.
    :param n: How many F->M transformations to perform. defaults is 1000.
    :return: a list containing: [number of matrices that follow constraint("good matrices"), total cost of good matrices,
             number of matrices that don't follow the constraints("bad matrices"), total cost of bad matrices].
    cost is calculated according to T->M transformation
    """
    good_mats, bad_mats = 0, 0  # counts how many t matrices follow/don't follow the constraint
    good_cost, bad_cost = 0, 0  # average cost for matrices that follow/ don't follow the constraint
    for i in range(n):
        F = Fst(f)
        t = F.produce_coalescence()
        T = Coalescence(t)
        cost = T.produce_migration()[1].cost
        if check_constraint(t):
            good_mats += 1
            good_cost += cost
        else:
            bad_cost += cost
            bad_mats += 1
    return [good_mats, good_cost, bad_mats, bad_cost]


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


def plots_1(n_matrices: 100, n_transformations: 1000, size=4) -> None:
    """
    plot that analyzes the T matrices in the F->T->M transformation.
    :param n_matrices: number of pseudo-random Fst matrices to generate fo analysis.
    :param n_transformations: number of transformations to generate from each Fst matrix
    :param size: size of each Fst matrix (number of populations).
    """
    prec_good_mats = []  # list of % of good T matrices for all F matrices generated
    avg_cost_bad = []  # list of avg costs of bad matrices
    avg_cost_good = []  # list of avg costs of good matrices
    matrices = np.array([i for i in range(1, n_matrices + 1)]).astype(int)
    for i in range(n_matrices):
        f = generate_pseudo_random_fst_mat(n=size)
        good_mats, good_cost, bad_mats, bad_cost = analyze_t_matrices(f, n_transformations)
        prec_good_mats.append(good_mats / n_transformations * 100)
        if bad_mats == 0:
            avg_cost_bad.append(0)
        else:
            avg_cost_bad.append(bad_cost / bad_mats)
        if good_mats == 0:
            avg_cost_good.append(0)
        else:
            avg_cost_good.append(good_cost / good_mats)
    avg_prec = np.mean(prec_good_mats)
    bad_cost_avg = np.mean(avg_cost_bad)
    good_cost_avg = np.mean(avg_cost_good)
    x_ticks = [20, 40, 60, 80]
    plt.figure(1)
    plt.plot(matrices, prec_good_mats, '-o', label=f"Average: {np.round(avg_prec, 2)}%")
    plt.xlabel('F matrix')
    plt.xticks(ticks=x_ticks)
    plt.ylabel('Good T matrices (%)')
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.plot(matrices, avg_cost_good, '-o', color="green",
             label=f"Good T matrices (Avg = {np.round(good_cost_avg, 2)})")
    plt.plot(matrices, avg_cost_bad, '-o', color="red", label=f"Bad T matrices (Avg = {np.round(bad_cost_avg, 2)})")
    # plt.annotate(f"Average: {np.round(bad_cost_avg, 2)}", (np.argmin(avg_cost_bad) + 1, np.min(avg_cost_bad)),
    #              textcoords="offset points", xytext=(0, -10), ha='center', size=8)
    # plt.annotate(f"Average: {np.round(good_cost_avg, 2)}", (np.argmax(avg_cost_good) + 1, np.max(avg_cost_good)),
    #              textcoords="offset points", xytext=(0, 10), ha='center', size=8)
    plt.xlabel('F matrix')
    plt.xticks(ticks=x_ticks)
    plt.ylabel("Average cost")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plots_1(100, 1000)
    plots_1(100, 1000, size=3)
    # plot_transformations(size=3)
    # for i in range(5):
    #     f = generate_pseudo_random_fst_mat(n=4)
    #     print(f, "\n")
    #     results = analyze_t_matrices(f)
    #     print(f"% of matrices that follow the constraint: {results[0]}\n")
    #     print(f"Average cost of the transformation T->M for T matrices that follow the constraint: {results[1]}\n")
    #     print(f"For T matrices that don't follow the constraint: {results[2]}\n")
