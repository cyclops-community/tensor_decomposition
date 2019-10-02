import numpy as np
import sys
import time


def init_rand(tenpy, order, s, R, sp_frac=1., seed=1):
    tenpy.seed(seed * 1001)
    A = []
    for i in range(order):
        A.append(tenpy.random((s, R)))
    if sp_frac < 1.:
        O = tenpy.sparse_random([s] * order, 1., 1., sp_frac)
        T = tenpy.TTTP(O, A)
    else:
        T = tenpy.ones([s] * order)
        T = tenpy.TTTP(T, A)
        O = None
    return [T, O]


def init_neg_rand(tenpy, order, s, R, sp_frac=1., seed=1):
    tenpy.seed(seed * 1001)
    A = []
    for i in range(order):
        A.append(-1 * tenpy.random((s, R)) + tenpy.random((s, R)))
    if sp_frac < 1.:
        O = tenpy.sparse_random([s] * order, 1., 1., sp_frac)
        T = tenpy.TTTP(O, A)
    else:
        T = tenpy.ones([s] * order)
        T = tenpy.TTTP(T, A)
        O = None
    return [T, O]


def collinearity(v1, v2, tenpy):
    return tenpy.dot(v1, v2) / (tenpy.vecnorm(v1) * tenpy.vecnorm(v2))


def init_collinearity_tensor(tenpy, s, order, R, col=[0.2, 0.8], seed=1):

    assert (col[0] >= 0. and col[1] <= 1.)
    assert (s >= R)
    tenpy.seed(seed * 1001)

    A = []
    for i in range(order):
        Gamma_L = tenpy.random((s, R))
        Gamma = tenpy.dot(tenpy.transpose(Gamma_L), Gamma_L)
        Gamma_min, Gamma_max = Gamma.min(), Gamma.max()
        Gamma = (Gamma - Gamma_min) / (Gamma_max - Gamma_min) * \
            (col[1] - col[0]) + col[0]
        tenpy.fill_diagonal(Gamma, 1.)
        A_iT = tenpy.cholesky(Gamma)
        # change size from [R,R] to [s,R]
        mat = tenpy.random((s, s))
        [U_mat, sigma_mat, VT_mat] = tenpy.svd(mat)
        A_iT = tenpy.dot(A_iT, VT_mat[:R, :])

        A.append(tenpy.transpose(A_iT))
        col_matrix = tenpy.dot(tenpy.transpose(A[i]), A[i])
        col_matrix_min, col_matrix_max = col_matrix.min(), (col_matrix - \
                                                        tenpy.eye(R, R)).max()
        assert (col_matrix_min - col[0] >= -0.01
                and col_matrix_max <= col[1] + 0.01)

    T = tenpy.ones([s] * order)
    T = tenpy.TTTP(T, A)
    O = None
    return [T, O]
