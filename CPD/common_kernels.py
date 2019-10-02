import numpy as np
import sys
import time


def compute_lin_sysN(tenpy, A, i, Regu):
    S = None
    for j in range(len(A)):
        if j != i:
            if S is None:
                S = tenpy.dot(tenpy.transpose(A[j]), A[j])
            else:
                S *= tenpy.dot(tenpy.transpose(A[j]), A[j])
    S += Regu * tenpy.eye(S.shape[0])
    return S


def compute_lin_sys(tenpy, X, Y, Regu):
    return tenpy.dot(tenpy.transpose(X), X) * tenpy.dot(
        tenpy.transpose(Y), Y) + Regu * tenpy.eye(Y.shape[1])


def compute_number_of_variables(G):
    a = 0
    for matrix in G:
        a += matrix.shape[0] * matrix.shape[1]
    return a


def flatten_Tensor(tenpy, G):
    l = compute_number_of_variables(G)
    g = tenpy.zeros(l)
    start = 0
    for i in range(len(G)):
        end = start + G[i].shape[0] * G[i].shape[1]
        g[start:end] = tenpy.reshape(G[i], -1, order='F')
        start = end
    return g


def reshape_into_matrices(tenpy, x, template):
    ret = []
    start = 0
    for i in range(len(template)):
        end = start + template[i].shape[0] * template[i].shape[1]
        A = tenpy.reshape(x[start:end], template[i].shape, order='F')
        ret.append(A)
        start = end
    return ret


def conjugate_gradient(A, x, b, tol=1e-4):
    tol = np.max([tol, tol * la.norm(b)])
    r = b - A.dot(x)
    if la.norm(r) < tol:
        return x
    p = r
    counter = 0
    while True:
        alpha = np.inner(r, r) / np.inner(p, A.dot(p))
        x += alpha * p
        r_new = r - alpha * A.dot(p)
        if la.norm(r_new) < tol:
            break
        beta = np.inner(r_new, r_new) / np.inner(r, r)
        p = r_new + beta * p
        r = r_new
        counter += 1
    return x, counter


def preconditioned_conjugate_gradient(A, x, b, M, tol=1e-4, formula="PR"):
    tol = np.max([tol, tol * la.norm(b)])
    r = b - A.dot(x)
    if la.norm(r) < tol:
        return x
    z = M.dot(r)
    p = z
    counter = 0
    while True:
        alpha = np.inner(r, z) / np.inner(p, A.dot(p))
        x += alpha * p
        r_new = r - alpha * A.dot(p)
        if la.norm(r_new) < tol:  ## need to add max iteration
            break
        z_new = M.dot(r_new)
        if formula == "PR":
            beta = np.inner(z_new, r_new - r) / np.inner(z, r)
        else:
            beta = np.inner(z_new, r_new) / np.inner(z, r)
        p = z_new + beta * p
        r = r_new
        z = z_new
        counter += 1
    return x, counter


def solve_sys_svd(tenpy, G, RHS):
    t0 = time.time()
    [U, S, VT] = tenpy.svd(G)
    S = 1. / S
    X = tenpy.dot(RHS, U)
    0. * X.i("ij") << S.i("j") * X.i("ij")
    t1 = time.time()
    tenpy.printf("Solving linear system took ", t1 - t0, "seconds")
    return tenpy.dot(X, VT)


def solve_sys(tenpy, G, RHS):
    L = tenpy.cholesky(G)
    X = tenpy.solve_tri(L, RHS, True, False, True)
    X = tenpy.solve_tri(L, X, True, False, False)
    return X


def get_residual(tenpy, T, A):
    t0 = time.time()
    V = tenpy.ones(T.shape)
    nrm = tenpy.vecnorm(T - tenpy.TTTP(V, A))
    t1 = time.time()
    tenpy.printf("Residual computation took", t1 - t0, "seconds")
    return nrm


def get_residual_sp(tenpy, O, T, A):
    t0 = time.time()
    K = tenpy.TTTP(O, A)
    nrm1 = tenpy.vecnorm(K)**2
    B = tenpy.dot(tenpy.transpose(A[0]), A[0])
    for i in range(1, len(A)):
        B *= tenpy.dot(tenpy.transpose(A[i]), A[i])
    nrm2 = tenpy.sum(B)
    diff = T - K
    nrm3 = tenpy.vecnorm(diff)**2
    nrm = (nrm3 + nrm2 - nrm1)**.5
    t1 = time.time()
    tenpy.printf("Sparse residual computation took", t1 - t0, "seconds")
    return nrm
