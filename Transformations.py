import matplotlib.pyplot as plt
# import numpy as np
# from Migration import Migration
# from Coalescence import Coalescence
# from Fst import Fst
# from Matrix_generator import generate_random_migration_mat


# def plot_M_to_F(size: int = 3, fig_num: int = 1) -> None:
#     """
#     generates a full transformation M->T->F from a random migration matrix, and plots all the matrices.
#     :param size: size of the matrices to generate.
#     :param fig_num: number of figure for plotting the matrices.
#     :return:
#     """
    # m = generate_random_migration_mat(decimals=3)
    # M = Migration(m)
    # t = M.produce_coalescence()
    # T = Coalescence(t)
    # f = T.produce_fst()
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    #data2D = np.random.random((50, 50))
    #plt.savefig('fig1')
x = [0,0.5,1]
y = 3
fig, ax = plt.subplots()
ax.plot(x, y)
//
#
# if __name__ == "__main__":
#     for i in range(3):
#         plot_M_to_F(size=3, fig_num=i)
