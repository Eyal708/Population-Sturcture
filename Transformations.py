import matplotlib.pyplot as plt
import numpy as np
from Migration import Migration
from Coalescence import Coalescence
from Fst import Fst
from Matrix_generator import generate_random_migration_mat


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
    new_t = original_F.produce_coalescence(bounds=(0, np.max(original_t)))
    new_T = Coalescence(new_t)
    new_m = new_T.produce_migration()
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
        fig[k].imshow(mat, cmap="plasma")
        # create a colorbar
        # ax[k].figure.colorbar(im, ax=ax)
        # add labels for each cell
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                fig[k].text(j, i, np.round(mat[i, j], 2), ha="center", va="center", color="black")
        fig[k].set_title(names[k])


if __name__ == "__main__":
    plot_transformations(size=4)
