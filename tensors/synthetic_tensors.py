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
    
    
def init_randn(tenpy, order, s, R, sp_frac=1., seed=1):
    tenpy.seed(seed * 1001)
    np.random.seed(seed * 1001)
    A = []
    for i in range(order):
        if tenpy.name() == 'ctf':
            A.append(tenpy.from_nparray(np.random.randn(s, R)))
        else:
            A.append(np.random.randn(s, R) )
    if sp_frac < 1.:
        O = tenpy.sparse_random([s] * order, 1., 1., sp_frac)
        T = tenpy.TTTP(O, A)
    else:
        T = tenpy.ones([s] * order)
        T = tenpy.TTTP(T, A)
        O = None
    return [T, O]

#def init_randn(tenpy, order, s, R, sp_frac=1., seed=1):
#    tenpy.seed(seed * 1001)
#    A = []
#    for i in range(order):
#        #if tenpy.name() == 'ctf':
#        n = 900
#        p = tenpy.sum(tenpy.random((s,R,n)),2)
#        p= ((p/n) - 0.5*tenpy.ones((s,R)))/((1/np.sqrt(12))/np.sqrt(n)) # (bar{X} - mu)/(sigma/root{n})
#        A.append(p)
            
 #       else:
 #           A.append(np.random.randn(s, R) )
#    if sp_frac < 1.:
#        O = tenpy.sparse_random([s] * order, 1., 1., sp_frac)
#        T = tenpy.TTTP(O, A)
#    else:
#        T = tenpy.ones([s] * order)
#        T = tenpy.TTTP(T, A)
#        O = None
#    return [T, O]


def init_rand3(tenpy, s, R, sp_frac=1.):
    A = tenpy.random((s, R))
    B = tenpy.random((s, R))
    C = tenpy.random((s, R))
    if sp_frac < 1.:
        O = tenpy.sparse_random((s, s, s), 1., 1., sp_frac)
        T = tenpy.TTTP(O, [A, B, C])
    else:
        T = tenpy.einsum("ia,ja,ka->ijk", A, B, C)
        O = tenpy.ones(T.shape)
    A = tenpy.random((s, R))
    B = tenpy.random((s, R))
    C = tenpy.random((s, R))
    return [A, B, C, T, O]


def init_mm(tenpy, s, R, seed=1):
    tenpy.seed(seed * 1001)
    sr = int(np.sqrt(s) + .1)
    assert(sr * sr == s)
    T = tenpy.tensor((sr, sr, sr, sr, sr, sr), sp=True)
    F = tenpy.tensor((sr, sr, sr, sr), sp=True)
    I = tenpy.speye(sr)
    #F.i("ijkl") << I.i("ik")*I.i("jl");
    # tenpy.einsum("ijab,klbc,mnca->ijklmn",F,F,F,out=T)
    T = tenpy.einsum("ia,jb,kb,lc,mc,na->ijklmn", I, I, I, I, I, I)
    T = T.reshape((s, s, s))
    O = T
    #A = tenpy.random((s,R))
    #B = tenpy.random((s,R))
    #C = tenpy.random((s,R))
    return [T, O]


def init_poisson(s, R):
    sr = int(s**(1. / 2) + .1)
    assert(sr * sr == s)
    T = tenpy.tensor((sr, sr, sr, sr, sr, sr), sp=True)
    A = (-2.) * tenpy.eye(sr, sr, sp=True) + tenpy.eye(sr,
                                                       sr, 1, sp=True) + tenpy.eye(sr, sr, -1, sp=True)
    I = tenpy.eye(sr, sr, sp=True)  # sparse identity matrix
    T.i("aixbjy") << A.i("ab") * I.i("ij") * I.i("xy") + I.i("ab") * \
        A.i("ij") * I.i("xy") + I.i("ab") * I.i("ij") * A.i("xy")
    # T.i("abijxy") << A.i("ab")*I.i("ij")*I.i("xy") + I.i("ab")*A.i("ij")*I.i("xy") + I.i("ab")*I.i("ij")*A.i("xy")
    N = tenpy.sparse_random((s, s, s), -0.000, .000, 1. / s)
    T = T.reshape((s, s, s)) + N
    [inds, vals] = T.read_local()
    vals[:] = 1.
    O = tenpy.tensor(T.shape, sp=True)
    O.write(inds, vals)
    A = tenpy.random((s, R))
    B = tenpy.random((s, R))
    C = tenpy.random((s, R))
    return [A, B, C, T, O]


def init_mom_cons(tenpy, k):
    order = 4
    mode_weights = [1, 1, -1, -1]

    delta = tenpy.tensor(k * np.ones(order))
    [inds, vals] = delta.read_local()
    new_inds = []
    for i in range(len(inds)):
        kval = 0
        ind = inds[i]
        iinds = []
        for j in range(order):
            ind_i = ind % k
            iinds.append(ind_i)
            ind = ind // k
            kval += mode_weights[-j] * ind_i
        if kval % k == 0:
            new_inds.append(inds[i])
    delta.write(new_inds, np.ones(len(new_inds)))
    return delta


def init_mom_cons_sv(tenpy, k):
    order = 4
    mode_weights = [1, 1, -1, -1]

    delta = tenpy.tensor(k * np.ones(order))
    [inds, vals] = delta.read_local()
    new_inds = []
    for i in range(len(inds)):
        kval = 0
        ind = inds[i]
        iinds = []
        for j in range(order):
            ind_i = ind % k
            iinds.append(ind_i)
            ind = ind // k
            kval += mode_weights[-j] * ind_i
        if kval % k == 0:
            new_inds.append(inds[i])
    delta.write(new_inds, np.ones(len(new_inds)))
    [U, S, VT] = delta.i("ijkl").svd("ija", "akl", threshold=1.e-3)
    return U


def collinearity(v1, v2, tenpy):
    return tenpy.dot(v1, v2) / (tenpy.vecnorm(v1) * tenpy.vecnorm(v2))


def init_collinearity_tensor(tenpy, s, order, R,
                             col=[0.2, 0.8],
                             seed=1):

    assert(col[0] >= 0. and col[1] <= 1.)
    assert(s >= R)
    tenpy.seed(seed * 1001)

    A = []
    for i in range(order):
        #Gamma_L = tenpy.random((s, R))
        Gamma_L = np.random.randn(s,R)
        Gamma = tenpy.dot(tenpy.transpose(Gamma_L), Gamma_L)
        Gamma_min, Gamma_max = Gamma.min(), Gamma.max()
        Gamma = (Gamma - Gamma_min) / (Gamma_max - Gamma_min) * \
            (col[1] - col[0]) + col[0]
        tenpy.fill_diagonal(Gamma, 1.)
        A_iT = tenpy.cholesky(Gamma)
        # change size from [R,R] to [s,R]
        #mat = tenpy.random((s, s))
        mat = np.random.randn(s,s)
        [U_mat, sigma_mat, VT_mat] = tenpy.svd(mat)
        A_iT = tenpy.dot(A_iT, VT_mat[:R, :])

        A.append(tenpy.transpose(A_iT))
        col_matrix = tenpy.dot(tenpy.transpose(A[i]), A[i])
        col_matrix_min, col_matrix_max = col_matrix.min(), (col_matrix - \
                                                        tenpy.eye(R, R)).max()
        assert(
            col_matrix_min -
            col[0] >= -
            0.01 and col_matrix_max <= col[1] +
            0.01)

    T = tenpy.ones([s] * order)
    T = tenpy.TTTP(T, A)
    O = None
    return [T, O]