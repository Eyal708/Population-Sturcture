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
    test_name2 = "Test 2: 3 populations according to motif 7, incoming migration rate is M=" + str(M)
    M_2 = Migration(test_mat2)
    result_2 = M_2.produce_coalescence()
    migration = 3 * (1 + 2 / (3 * M))
    T_2 = np.array([[3, migration, migration], [migration, 3, migration], [migration, migration, 3]])
    test_2 = [test_name2, M_2, test_mat2, result_2, T_2]
    # test 3: 3 populations where two are not connected (motif 6)
    test_mat3 = np.array([[0, 1, 1], [2, 0, 0], [2, 0, 0]])
    M = np.sum(test_mat3[0, :])
    test_name3 = "Test 3: 3 populations, according to motif 6, incoming migration rate is M=" + str(M)
    M_3 = Migration(test_mat3)
    result_3 = M_3.produce_coalescence()
    t1_2 = 8 / 3 * (1 + 5 / (8 * M))
    t1_3 = t1_2
    t2_3 = 8 / 3 * (1 + 1 / M)
    T_3 = np.array([[8 / 3, t1_2, t1_3], [t1_2, 8 / 3, t2_3], [t1_3, t2_3, 8 / 3]])
    test_3 = [test_name3, M_3, test_mat3, result_3, T_3]

    # test 4:  3 populations, one isolated
    # test_mat_4 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    # M = np.sum(test_mat_4[1, :])
    # test_name4 = "Test 4: 3 populations, one isolated, incoming migrants rate M=" + str(M)
    # t2_3 = 2 * (1 + 1 / (2 * M))
    # T_4 = np.array([[1, np.inf, np.inf], [np.inf, 2, t2_3], [np.inf, t2_3, 2]])
    # M_4 = Migration(test_mat_4)
    # result_4 = M_4.produce_coefficient_matrix()
    # print(np.linalg.matrix_rank(result_4))
    # print(result_4)
    # test 5: two populations
    test_mat5 = np.array([[0, 1], [1, 0]])
    M = np.sum(test_mat5[0, :])
    test_name5 = "Test 5: 2 connected populations (motif 2),incoming migration rate is M=" + str(M)
    t1_2 = 2 * (1 + 1 / (2 * M))
    T_5 = np.array([[2, t1_2], [t1_2, 2]])
    M_5 = Migration(test_mat5)
    result_5 = M_5.produce_coalescence()
    test_5 = [test_name5, M_5, test_mat5, result_5, T_5]

    # test 6: 4 populations- motif 13
    test_mat6 = np.array([[0, 1, 1, 1], [3, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0]])
    M = np.sum(test_mat6[0, :])
    test_name6 = "Test 6: 4 populations according to motif 13, migration rate is M=" + str(M)
    t_diagonal = 3
    t1_2 = 3 * (1 + 2 / (3 * M))
    t1_3, t1_4 = t1_2, t1_2
    t2_3 = 3 * (1 + (1 / M))
    t2_4, t3_4 = t2_3, t2_3
    T_6 = np.array([[t_diagonal, t1_2, t1_3, t1_4], [t1_2, t_diagonal, t2_3, t2_4], [t1_3, t2_3, t_diagonal, t3_4],
                    [t1_4, t2_4, t3_4, t_diagonal]])
    M_6 = Migration(test_mat6)
    result_6 = M_6.produce_coalescence()
    test_6 = [test_name6, M_6, test_mat6, result_6, T_6]

    # test 7: 4 populations according to motif 14
    test_mat7 = np.array([[0, 2, 0, 0], [1, 0, 0, 1], [0, 0, 0, 2], [0, 1, 1, 0]])
    M = np.sum(test_mat7[0, :])
    test_name7 = "Test 7: 4 populations according to motif 14, incoming migration rate is M=" + str(M)
    t1_1 = 2 * (45 * M + 44) / (25 * M + 28)
    t2_2 = 2 * (45 * M + 52) / (25 * M + 28)
    t4_4 = t2_2
    t3_3 = t1_1
    t1_2 = (3 * (30 * M ** 2 + 51 * M + 20)) / (M * (25 * M + 28))
    t1_3 = (90 * M ** 2 + 241 * M + 156) / (M * (25 * M + 28))
    t1_4 = (2 * (3 * M + 4) * (15 * M + 16)) / (M * (25 * M + 28))
    t2_3 = t1_4
    t2_4 = (90 * M ** 2 + 185 * M + 92) / (M * (25 * M + 28))
    t3_4 = t1_2
    T_7 = np.array(
        [[t1_1, t1_2, t1_3, t1_4], [t1_2, t2_2, t2_3, t2_4], [t1_3, t2_3, t3_3, t3_4], [t1_4, t2_4, t3_4, t4_4]])
    M_7 = Migration(test_mat7)
    result_7 = M_7.produce_coalescence()
    test_7 = [test_name7, M_7, test_mat7, result_7, T_7]

    tests = [test_1, test_2, test_3, test_5, test_6, test_7]
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
            print("Passed " + test[0] + "\n")
        print(".......................................................................\n")


if __name__ == "__main__":
    test_produce_coalescence()