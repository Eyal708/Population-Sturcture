import numpy as np
import matplotlib.pyplot as plt
from Matrix_generator import generate_pseudo_random_fst_mat
from Coalescence import Coalescence
from Fst import Fst
import time
import pickle


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
    :param n_matrices: number of pseudo-random Fst matrices to generate for analysis.
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
    plt.xlabel('F matrix')
    plt.xticks(ticks=x_ticks)
    plt.ylabel("Average cost")
    plt.legend()
    plt.show()


def generate_and_save_transformations(f: np.ndarray, n_transformations: int, path: str) -> int:
    """
    Generates F->T>M transformations and saves it as a pickle file. The created pickle file contains a dictionary
    with 3 keys: 'f', 't', 'm','cost'. The value of key 'f' is the initial F matrix. The values of each 't'
    and 'm' keys is a list with all the matrices generated, where the index of the M matrix corresponds to the index
    of the T matrix it was generated from. The value of the 'cost' key is a list with all the costs of T->M
    transformations. Only T matrices that follow the within < inbetween constraint are used.
    :param f: Initial Fst matrix.
    :param n_transformations: number of F->T transformation to perform. Only T matrices that follow the constraint
    are used for the T->M transformation, so the final number of matrices produced is unknown.
    :param path: path for the saved pickle file.
    :return: Number of M matrices produced, which is the number of T matrices that followed the constraint.
    """
    F = Fst(f)
    mat_dict = {'f': f, 't': [], 'm': [], 'cost': []}
    good_mats = 0
    for i in range(n_transformations):
        t = F.produce_coalescence()
        if check_constraint(t):
            T = Coalescence(t)
            result = T.produce_migration()
            m, cost = result[0], result[1].cost
            mat_dict['t'].append(t)
            mat_dict['m'].append(m)
            mat_dict['cost'].append(cost)
            good_mats += 1
    pickle_file = open(path, 'ab')
    pickle.dump(mat_dict, pickle_file)
    pickle_file.close()
    return good_mats


def store_transformations(shape: int, n_matrices: int, n_transformations: int) -> None:
    """
    Stores F->T->M transformations from pseudo random matrices as pickle files.
    :param shape: shape of the matrices to generate
    :param n_matrices: How many F matrices to produce
    :param n_transformations: How many transformations to generate from each F matrix. Only transformations where
    T follows the constraint are saved.
    """
    for i in range(n_matrices):
        f = generate_pseudo_random_fst_mat(n=shape)
        generate_and_save_transformations(f, n_transformations, f"pickles/{shape}X{shape}_transformation_{i + 1}")


if __name__ == "__main__":
    # store_transformations(shape=3, n_matrices=100, n_transformations=1000)
    # store_transformations(shape=5, n_matrices=100, n_transformations=1000)
    file = open("pickles/5X5_transformation_94",'rb')
    mats = pickle.load(file)
    print(np.round(mats['t'][3],2))
    print(np.round(mats['m'][3],2))
    # plots_1(100, 1000)
    # start = time.time()
    # plots_1(100, 1000, size=5)
    # end = time.time()
    # print(f"Running time is {end - start} seconds")
