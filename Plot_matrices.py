import matplotlib.pyplot as plt
import numpy as np
from migration import Migration
from coalescence import Coalescence
from fst import Fst
from matrix_generator import generate_random_migration_mat


def plot_5_transformations(initial_m: np.ndarray) -> None:
    """
    plots 5 possible F->T'->M' transformations, induced by an original M->T->F transformation.
    :param initial_m: initial migration matrix from which we induce T and F matrices.
    """
    initial_M = Migration(initial_m)
    initial_t = initial_M.produce_coalescence()
    initial_T = Coalescence(initial_t)
    initial_f = initial_T.produce_fst()
    initial_F = Fst(initial_f)
    # create figure for initial matrices
    fig_1, ax_1 = plt.subplots(nrows=1, ncols=3)
    add_matrices_to_plot([initial_m, initial_t, initial_f], ["M", "T", "F"], ax_1)
    fig_1.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_2, ax_2 = plt.subplots(nrows=3, ncols=5)
    t_matrices, m_matrices = [], []
    for i in range(5):
        new_t = initial_F.produce_coalescence()
        new_T = Coalescence(new_t)
        new_m = new_T.produce_migration()[0]
        t_matrices.append(new_t)
        m_matrices.append(new_m)
    add_matrices_to_plot([initial_f], ["F"], np.array([ax_2[0, 2]]))
    add_matrices_to_plot(t_matrices, ["", "", "T", "", ""], ax_2[1, :])
    add_matrices_to_plot(m_matrices, ["", "", "M", "", ""], ax_2[2, :])
    for col in range(5):
        if col != 2:
            fig_2.delaxes(ax_2[0, col])
    fig_2.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_2.tight_layout()
    plt.show()


def plot_transformations(size: int = 3) -> None:
    """
    plots a full transformation M -> T -> F from a randomly generated migration matrix (first row of the plot), and on
    the second row, plots a possible reverse transformation F -> T' -> M'.
    :param size: size of the migration matrix to generate(will be the size of all matrices).
    """
    original_m = generate_random_migration_mat(n=size)
    original_M = Migration(original_m)
    original_t = original_M.produce_coalescence()
    original_T = Coalescence(original_t)
    original_f = original_T.produce_fst()
    original_F = Fst(original_f)
    new_t = original_F.produce_coalescence()
    new_T = Coalescence(new_t)
    new_m = new_T.produce_migration()[0]
    fig, ax = plt.subplots(nrows=2, ncols=3)
    add_matrices_to_plot([original_m, original_t, original_f], ["M", "T", "F"], ax[0, :])
    add_matrices_to_plot([original_f, new_t, new_m], ["F", "T", "M"], ax[1, :])
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()


def add_matrices_to_plot(matrices: list[np.ndarray], names: list[str], fig: np.ndarray) -> None:
    """"
    Adds matrices to a figure as subplots
    :param fig: array containing subplots
    :param matrices: matrices to add to plot
    :param names: names of matrices to add

    """
    for k, mat in enumerate(matrices):
        im = fig[k].imshow(mat, cmap="Oranges")
        # create a colorbar
        # if k == 1:
        #     im.figure.colorbar(im)
        # add labels for each cell
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                fig[k].text(j, i, np.round(mat[i, j], 2), ha="center", va="center", color="black")
        fig[k].set_title(names[k])
        fig[k].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)




if __name__ == "__main__":
    plot_transformations(size=5)
    #plot_5_transformations(generate_random_migration_mat(n=3))
