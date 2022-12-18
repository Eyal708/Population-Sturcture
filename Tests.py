import numpy as np
from Migration import Migration


def test_produce_coalescence() -> None:
    """
    test the produce_coalescence function of the Migration class
    """
    # test 1: random matrix
    test_name1 = "Test 1: Random matrix"
    test_mat1 = np.array([[0, 0.3, 0.2], [0.1, 0, 0.4], [0.4, 0.2, 0]])
    M_1 = Migration(test_mat1)
    result_1 = M_1.produce_coalescence()
    T_1 = result_1
    test_1 = [test_name1, M_1, test_mat1, result_1, T_1]

    # test 2: motif of 3 connected populations with equal migration
    test_mat2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    M = np.sum(test_mat2[0, :])  # Total migration from each population
    test_name2 = "Test 2: 3 populations: fully connected, symmetric and equal migration M=" + str(M)
    M_2 = Migration(test_mat2)
    result_2 = M_2.produce_coalescence()
    migration = 3 * (1 + 2 / (3 * M))
    T_2 = np.array([[3, migration, migration], [migration, 3, migration], [migration, migration, 3]])
    test_2 = [test_name2, M_2, test_mat2, result_2, T_2]
    # test 3: 3 populations where two are not connected (motif 6)
    test_mat3 = np.array([[0, 1, 1], [2, 0, 0], [2, 0, 0]])
    M = np.sum(test_mat3[0, :])
    test_name3 = "Test 3: 3 populations, 2 not connected symmetric,incoming migrants M=" + str(M)
    M_3 = Migration(test_mat3)
    result_3 = M_3.produce_coalescence()
    t1_2 = 8 / 3 * (1 + 5 / (8 * M))
    t1_3 = t1_2
    t2_3 = 8 / 3 * (1 + 1 / M)
    T_3 = np.array([[8 / 3, t1_2, t1_3], [t1_2, 8 / 3, t2_3], [t1_3, t2_3, 8 / 3]])
    test_3 = [test_name3, M_3, test_mat3, result_3, T_3]
    tests = [test_1, test_2, test_3]
    for num, test in enumerate(tests):
        print("Performing " + test[0] + "\nMigration matrix: \n", test[2])
        A_test = test[1].produce_coefficient_matrix()  # coefficient matrix
        A_rank = np.linalg.matrix_rank(A_test)  # coefficient matrix rank
        print("The corresponding coefficients matrix is:\n", A_test, "\n")
        print("The coefficients matrix rank is:", A_rank, "\n")
        print("The calculated unique corresponding T matrix is:\n", test[3], "\n")
        if not np.array_equal(test[3].round(decimals=6), test[4].round(decimals=6)):
            print("Failed " + test[0] + "\nThe expected Coalescence matrix is:\n", test[4], "\n")
        else:
            print("Passed Test " + test[0] + "\n")
        print(".......................................................................\n")


if __name__ == "__main__":
    test_produce_coalescence()
