import math

import numpy as np
import matplotlib.pyplot as plt
from Matrix_generator import generate_random_migration_mat, generate_pseudo_random_fst_mat
from Coalescence import Coalescence
from Fst import Fst
from Migration import Migration
import time
import pickle
import seaborn as sb
import sys
from Helper_funcs import check_constraint, diameter, matrix_distance, matrix_mean


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


def generate_and_save_transformations(m: np.ndarray, n_transformations: int, path: str) -> int:
    """
    Generates F->T>M transformations and saves it as a pickle file. The created pickle file contains a dictionary
    with 8 keys:
    1) 'true_m': the migration matrix from which the F matrix was generated, meaning it is the 'true'
    underlying migration matrix.
    2) 'f': the starting F matrix.
    3) 'good_t': list of good T matrices generated from f.
    4) 'bad_t': list of bad T matrices generated from f.
    5) 'good_m': list of M matrices generated from the good T matrices.
    6) 'bad_m': list of M matrices generated from the bad T matrices.
    7) 'good_cost': list of costs of the T->M transformation for the good T matrices.
    8) 'bad_cost': list of costs of the T->M transformation for the bad T matrices.
    :param m: Initial M matrix from which the F matrix is generated.
    :param n_transformations: number of F->T transformation to perform. Only T matrices that follow the constraint
    are used for the T->M transformation, so the final number of matrices produced is unknown.
    :param path: path for the saved pickle file.
    :return: Number of good T matrices produced from f.
    """
    M = Migration(m)
    T = Coalescence(M.produce_coalescence())
    f = np.round(T.produce_fst(), decimals=2)
    F = Fst(f)
    mat_dict = {'true_m': m, 'f': f, 'good_t': [], 'bad_t': [], 'good_m': [], 'bad_m': [], 'good_cost': [],
                'bad_cost': []}
    good_mats = 0
    for i in range(n_transformations):
        t = F.produce_coalescence()
        T = Coalescence(t)
        result = T.produce_migration()
        m, cost = result[0], result[1].cost
        if check_constraint(t):
            mat_dict['good_t'].append(t)
            mat_dict['good_m'].append(m)
            mat_dict['good_cost'].append(cost)
            good_mats += 1
        else:
            mat_dict['bad_t'].append(t)
            mat_dict['bad_m'].append(m)
            mat_dict['bad_cost'].append(cost)

    pickle_file = open(path, 'ab')
    pickle.dump(mat_dict, pickle_file)
    pickle_file.close()
    return good_mats


def store_transformations(shape: int, n_matrices: int, n_transformations: int, dir_path: str) -> None:
    """
    Stores F->T->M transformations from pseudo random matrices as pickle files.
    :param dir_path: path to directory to save the pickle files.
    :param shape: shape of the matrices to generate.
    :param n_matrices: How many F matrices to produce.
    :param n_transformations: How many transformations to generate from each F matrix.
    """
    for i in range(n_matrices):
        m = generate_random_migration_mat(n=shape)
        generate_and_save_transformations(m, n_transformations, f"{dir_path}/{shape}X{shape}_transformation_{i + 1}")


def box_plot_pct():
    sizes = np.array([[3], [4], [5]])
    categories = np.repeat(sizes, 100)
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"fixed_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            num_t = len(mats['good_t'])
            data.append(num_t / 1000 * 100)
    plt.figure(figsize=(10, 14))
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_xlabel('Number of populations', fontsize=36)
    fig.set_ylabel('Good T matrices(%)', fontsize=36)
    fig.tick_params(axis='both', which='major', labelsize=26)
    # fig.set_yticks(range(0, 50, 10))
    plt.savefig("NewPlots/boxplot_pct.svg")
    plt.show()


def box_plot_good_vs_bad():
    categories = []
    data = []
    types = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"fixed_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            good_cost = np.array(mats['good_cost']) / (i ** 2 - i)  # normalize costs
            bad_cost = np.array(mats['bad_cost']) / (i ** 2 - i)  # normalize costs
            types.extend(['Good T matrices'] * len(good_cost))
            data.extend(good_cost)
            types.extend(['Bad T matrices'] * len(bad_cost))
            data.extend(bad_cost)
            categories.extend([i] * (len(good_cost) + len(bad_cost)))

    colors = {'Good T matrices': 'lightgreen', 'Bad T matrices': 'tomato'}
    palette = [colors[typ] for typ in ['Good T matrices', "Bad T matrices"]]
    # create the boxplot with grouped boxes
    plt.figure(figsize=(10, 14))
    ax = sb.boxplot(x=categories, y=data, hue=types, palette=palette, showfliers=False)

    # set y-axis ticks

    # label the axes and adjust font size
    ax.set_xlabel('Number of populations', fontsize=36)
    ax.set_ylabel('LS cost', fontsize=36)
    ax.axvline(x=0.5, linestyle='--', color='gray', linewidth=1)
    ax.axvline(x=1.5, linestyle='--', color='gray', linewidth=1)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)

    # adjust font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.savefig("NewPlots/boxplot_good_VS_bad.svg")
    # show the plot
    plt.show()


def box_plot_good_vs_distance():
    categories = []
    data = []
    types = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"fixed_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            true_m = mats["true_m"]
            good_d = [matrix_distance(mat, true_m) for mat in mats["good_m"]]
            bad_d = [matrix_distance(mat, true_m) for mat in mats["bad_m"]]
            types.extend(['Good M matrices'] * len(good_d))
            data.extend(good_d)
            types.extend(['Bad M matrices'] * len(bad_d))
            data.extend(bad_d)
            categories.extend([i] * (len(good_d) + len(bad_d)))

    colors = {'Good M matrices': 'lightgreen', 'Bad M matrices': 'orangered'}
    palette = [colors[typ] for typ in ['Good M matrices', "Bad M matrices"]]
    # create the boxplot with grouped boxes
    plt.figure(figsize=(10, 14))
    ax = sb.boxplot(x=categories, y=data, hue=types, palette=palette, showfliers=False)

    # set y-axis ticks

    # label the axes and adjust font size
    ax.set_xlabel('Number of populations', fontsize=36)
    ax.set_ylabel('Distance from ground truth', fontsize=36)
    ax.axvline(x=0.5, linestyle='--', color='gray', linewidth=1)
    ax.axvline(x=1.5, linestyle='--', color='gray', linewidth=1)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)

    # adjust font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.savefig("NewPlots/good_VS_bad_distance_all.svg")
    # show the plot
    plt.show()


def good_minimal_cost():
    """
    plots the average minimal cost of M matrices produced from good T matrices
    """
    sizes = np.array([[3], [4], [5]])
    categories = np.repeat(sizes, 100)
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            data.append(np.min(mats['good_cost']))
    plt.figure(figsize=(10, 14))
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_xlabel('Number of populations(n)', fontsize=26)
    fig.set_ylabel('Minimal cost', fontsize=26)
    fig.tick_params(axis='both', which='major', labelsize=22)
    # fig.set_yticks(range(0, 50, 10))
    plt.show()


def good_diameter_m():
    """
    plots the diameter of M matrices produced from good T matrices
    """
    sizes = np.array([[3], [4], [5]])
    categories = np.repeat(sizes, 100)
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"fixed_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            costs = np.array(mats['bad_cost']) / (i ** 2 - i)
            M_mats = np.array(mats["bad_m"])
            k = min(100,len(M_mats)-1) #int(np.ceil(costs.shape[0] / 100 * 10))  # find the 10% lowest cost
            indices = np.argpartition(costs, k)
            best_mats = M_mats[indices[:k]]
            data.append(diameter(best_mats))
    plt.figure(figsize=(10, 14))
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_title('Diameter of M matrices', fontsize=28)
    fig.set_xlabel('Number of populations(n)', fontsize=26)
    fig.set_ylabel('Diameter', fontsize=26)
    fig.tick_params(axis='both', which='major', labelsize=22)
    # fig.set_yticks(range(0, 50, 10))
    # plt.savefig("NewPlots/10%_good_diameter.svg")
    plt.show()


def good_diameter_t():
    """
     plots the diameter of good T matrices
     """
    sizes = np.array([[3], [4], [5]])
    categories = np.repeat(sizes, 100)
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            data.append(diameter(mats['good_t']))
    plt.figure(figsize=(10, 14))
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_title('Diameter of good T matrices', fontsize=28)
    fig.set_xlabel('Number of populations(n)', fontsize=26)
    fig.set_ylabel('Diameter', fontsize=26)
    fig.tick_params(axis='both', which='major', labelsize=22)
    # fig.set_yticks(range(0, 50, 10))
    plt.savefig("Plots/good_diameter_T.svg")
    plt.show()


def diameter_convergence(shape: int):
    """
    plot diameter convergence for good and bad T matrices
    """
    repeats = [i * 10 for i in range(0, 101)]
    good_diam, bad_diam = [0], [0]
    for n in repeats[1:]:
        n_good_diams, n_bad_diams = [], []
        for i in range(100):
            file = open(f"new_pickles/{shape}X{shape}_transformation_{i + 1}", 'rb')
            mats = pickle.load(file)
            good_mats, bad_mats = mats['good_m'], mats['bad_m']
            if len(good_mats) >= n:
                n_good_diams.append(diameter(good_mats[0:n]))
            if len(bad_mats) >= n:
                n_bad_diams.append(diameter(bad_mats[0:n]))
        good_diam.append(np.mean(n_good_diams))
        bad_diam.append((np.mean(n_bad_diams)))
    plt.plot(repeats, good_diam, color="green", label="Good T matrices")
    plt.plot(repeats, bad_diam, color="red", label=f"Bad T matrices")
    plt.title(f'{shape}X{shape} diameter convergence')
    plt.xlabel('Number of matrices')
    plt.ylabel("Diameter")
    plt.legend()
    plt.savefig(f"Plots/{shape}X{shape}_diam_conv.svg")
    plt.show()


def k_smallest_cost() -> None:
    """ find 10% best costs out of good T matrices. costs are normalized"""
    sizes = np.array([[3], [4], [5]])
    categories = np.repeat(sizes, 100)
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"new_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            costs = np.array(mats['good_cost']) / (i ** 2 - i)
            k = int(np.floor(costs.shape[0] / 100 * 10))  # find the 10% lowest cost
            indices = np.argpartition(costs, k)
            result = costs[indices[:k]]
            data.append(np.mean(result))
    plt.figure(figsize=(10, 14))
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_xlabel('Number of populations(n)', fontsize=26)
    fig.set_ylabel(f'Top 10% average cost (normalized)', fontsize=26)
    fig.tick_params(axis='both', which='major', labelsize=22)
    # fig.set_yticks(range(0, 50, 10))
    plt.show()


def plot_matrix_range(n: int):
    file = open(f"fixed_pickles/{n}X{n}_transformation_2", 'rb')
    mats = pickle.load(file)
    costs = np.array(mats['good_cost']) / (n ** 2 - n)
    M_mats = np.array(mats["good_m"])
    k = int(np.ceil(costs.shape[0] / 100 * 10))  # find the 10% lowest cost
    indices = np.argpartition(costs, k)
    best_mats = M_mats[indices[:k]]
    best_costs = costs[indices[:k]]
    diam = diameter(best_mats)
    best_mat = matrix_mean(best_mats)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    print(diam)
    matrices = [best_mat, mats["true_m"]]
    names = ["Inferred matrix", "Real Matrix"]
    for k in [0, 1]:
        ax[k].imshow(matrices[k], cmap="Oranges")
        for i in range(best_mat.shape[0]):
            for j in range(best_mat.shape[1]):
                ax[k].text(j, i, np.round(matrices[k][i, j], 2), ha="center", va="center", color="black")
        ax[k].set_title(names[k])
        ax[k].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig.show()


def inferred_mat_distance(which: str, good=True):
    # sizes = np.array([[3], [4], [5]])
    categories = []
    data = []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"Fixed_pickles_constraints/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            if good:
                costs = np.array(mats['good_cost']) / (i ** 2 - i)
                M_mats = np.array(mats["good_m"])
            else:
                costs = np.array(mats['bad_cost']) / (i ** 2 - i)
                M_mats = np.array(mats["bad_m"])

            k = int(np.ceil(costs.shape[0] / 100 * 10))  # find the 10% lowest cost
            indices = np.argpartition(costs, k)
            best_mats = M_mats[indices[:k]]
            best_costs = costs[indices[:k]]
            if which == "best":
                if len(best_mats) == 0:
                    break
                best_mat = best_mats[np.argmin(best_costs)]
            elif which == "random":
                best_mat = generate_random_migration_mat(i)
            else:
                if len(best_mats) == 0:
                    break
                best_mat = matrix_mean(best_mats)
            true_mat = mats["true_m"]
            data.append(matrix_distance(best_mat, true_mat))
            categories.append(i)
    name = "Distance from ground truth"
    plt.figure(figsize=(10, 14))
    # if not good:
    #     plt.title("Analysis for bad T matrices", fontsize=30)
    # else:
    #     plt.title("Analysis for good T matrices", fontsize=30)
    fig = sb.boxplot(x=categories, y=data, showfliers=False, palette='Set2')
    fig.set_xlabel('Number of populations', fontsize=36)
    fig.set_ylabel(name, fontsize=36)
    fig.set_yticks([0, 0.2, 0.4, 0.6, 0.8,1,1.2, 1.4])
    fig.tick_params(axis='both', which='major', labelsize=26)
    # fig.set_yticks(range(0, 50, 10))
    plt.savefig(f"NewPlots/Only_good_distance_{which}.svg")
    plt.show()


def compare_good_bad(which: str):
    categories, data, types = [], [], []
    for i in range(3, 6):
        for j in range(100):
            file = open(f"fixed_pickles/{i}X{i}_transformation_{j + 1}", 'rb')
            mats = pickle.load(file)
            good_costs = np.array(mats['good_cost']) / (i ** 2 - i)
            good_M_mats = np.array(mats["good_m"])
            bad_costs = np.array(mats['bad_cost']) / (i ** 2 - i)
            bad_M_mats = np.array(mats["bad_m"])
            good_k = int(np.ceil(good_costs.shape[0] / 100 * 10))  # find the 10% lowest cost
            good_indices = np.argpartition(good_costs, good_k)
            good_best_mats = good_M_mats[good_indices[:good_k]]
            good_best_costs = good_costs[good_indices[:good_k]]
            bad_k = int(np.ceil(bad_costs.shape[0] / 100 * 10))  # find the 10% lowest cost
            bad_indices = np.argpartition(bad_costs, bad_k)
            bad_best_mats = bad_M_mats[bad_indices[:bad_k]]
            bad_best_costs = bad_costs[bad_indices[:bad_k]]
            true_mat = mats["true_m"]
            if len(good_best_mats) > 0:
                types.append("Good T matrices")
                categories.append(i)
                if which == "best":
                    best_mat = good_best_mats[np.argmin(good_best_costs)]
                else:
                    best_mat = matrix_mean(good_best_mats)
                data.append(matrix_distance(best_mat, true_mat))
            if len(bad_best_mats) > 0:
                types.append("Bad T matrices")
                categories.append(i)
                if which == "best":
                    best_mat = bad_best_mats[np.argmin(bad_best_costs)]
                else:
                    best_mat = matrix_mean(bad_best_mats)
                data.append(matrix_distance(best_mat, true_mat))

    colors = {'Good T matrices': 'green', 'Bad T matrices': 'red'}
    palette = [colors[typ] for typ in ['Good T matrices', "Bad T matrices"]]
    # create the boxplot with grouped boxes
    plt.figure(figsize=(10, 14))
    ax = sb.boxplot(x=categories, y=data, hue=types, palette=palette, showfliers=False)

    # set y-axis ticks

    # label the axes and adjust font size
    ax.set_xlabel('Number of populations(n)', fontsize=36)
    ax.set_ylabel(f'Distance of true matrix from {which} matrix', fontsize=36)
    ax.axvline(x=0.5, linestyle='--', color='gray', linewidth=1)
    ax.axvline(x=1.5, linestyle='--', color='gray', linewidth=1)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)

    # adjust font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=26)
    # show the plot
    plt.savefig(f"NewPlots/good_VS_bad_distance_{which}.svg")
    plt.show()


if __name__ == "__main__":
    # box_plot_pct()
    box_plot_good_vs_bad()
    # box_plot_good_vs_distance()
    # compare_good_bad(which="best")
    # compare_good_bad(which="average")
    # inferred_mat_distance(which="avg", good=False)
    # inferred_mat_distance(which="avg")
    # inferred_mat_distance(which="best", good=False)
    # inferred_mat_distance(which="best")
    # inferred_mat_distance(which="avg")
    # inferred_mat_distance(which="random")
    # plot_matrix_range(5)
# k_smallest_cost()
# diameter_convergence(shape=3)
# good_diameter_t()
 #good_diameter_m()
# good_minimal_cost()
# store_transformations(shape=int(sys.argv[1]), n_matrices=100, n_transformations=1000, dir_path='new_pickles')
# store_transformations(shape=5, n_matrices=100, n_transformations=1000)
# file = open("new_pickles/3X3_transformation_1", 'rb')
# mats = pickle.load(file)
# costs = mats['bad_cost']
# print(mats['true_m'])
# print(costs)
# plots_1(100, 1000)
# start = time.time()
# plots_1(100, 1000, size=5)
# end = time.time()
# print(f"Running time is {end - start} seconds")
