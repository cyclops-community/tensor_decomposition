import numpy as np
import sys
import time

import numpy.linalg as la
#import error_computation
from scipy.linalg import svd


def generate_cov_with_collinearity(tenpy, n, collinearity_level=0.9, rank=None, seed=1):
    
    tenpy.seed(seed * 1001)

    base_vector = np.random.randn(n)
    V = []
    for _ in range(n if rank is None else rank):
        noise = np.random.randn(n) * (1 - collinearity_level)  # Add small noise
        correlated_vector = collinearity_level * base_vector + noise
        V.append(correlated_vector)
    
    V = np.array(V).T  # Shape (n x rank) if low-rank, else (n x n)
    
    cov = V @ V.T
    cov += np.eye(n) * 1e-6  # Add small diagonal for numerical stability
    
    return cov

def generate_cov(n, alpha):
        #v = np.random.randint(1, n+1, size=n)  # Random integers from 1 to n
        #s = 1.5 ** -v 
        #s = np.sort(s)[::-1]
        s = alpha**(-np.linspace(0,n,n))
        A = np.random.random((n,n))
        Q, R = la.qr(A)
        cov = Q.T @ np.diag(s) @ Q
        #reg_strength = 0.5
        #cov += reg_strength* np.eye(n) #Regularized Covariance Matrix
        return cov
######  old model   
def generate_tensor_with_noise_old_model(tenpy, dims, R, k, epsilon, alpha, seed=1):
    #Generate a synthetic tensor T with additive noise based on random normal vectors.
    tenpy.seed(seed * 1001)
    np.random.seed(seed * 1001)
    
    N = len(dims)
    means = [tenpy.zeros(i) for i in dims]
    covs = [generate_cov(i, alpha) for i in dims]# Generate covariance matrices for each mode (dimension)
    #covs = [generate_cov_with_collinearity(tenpy, i, collinearity_level=0.9, rank=None, seed=1) for i in dims]# Generate covariance matrices for each mode (dimension)
    ###
    covs_inv = [la.pinv(c) for c in covs]
    M_true_inv = covs_inv
    # M_true_inv = covs_inv[0]
    # for i in range(1, N):
    #     M_true_inv = np.kron(M_true_inv, covs_inv[i])

    #M_true_inv = np.kron(np.kron(covs_inv[0],covs_inv[1]),covs_inv[2])
    ###
    T = tenpy.zeros(dims)
    U = []
    U = [np.random.multivariate_normal(means[i], covs[i], R).T for i in range(N)] 
    T = T + tenpy.einsum('ir,jr,kr->ijk', *U) # Generate tensor T
    T /= R  
    ###
    # Compute empirical covariance matrices for each mode
    covs_empirical = [samples @ samples.T for samples in U]
    #covs_empirical = [C/R for C in covs_empirical]
    sample_pinv = [la.pinv(samples) for samples in U]
    covs_pinv_empirical = [samples_pinv.T @ samples_pinv for samples_pinv in sample_pinv]
    # M_empirical_pinv = covs_pinv_empirical[0]
    # for i in range(1, N):
    #     M_empirical_pinv = np.kron(M_empirical_pinv, covs_pinv_empirical[i])

    #M_empirical_pinv = np.kron(np.kron(covs_pinv_empirical[0],covs_pinv_empirical[1]),covs_pinv_empirical[2])
    M_empirical_pinv = covs_pinv_empirical
    ###
    # Generate factor matrices for noise
    U_tilde = []
    for i in range(N):
       U_i = np.random.multivariate_normal(means[i], covs[i], k).T  # Random normal matrices of size n[i] x k
       U_i -= U_i.mean(axis=0)  # Subtract mean to get zero-mean distribution
       U_tilde.append(U_i)
    # Generate tensor N
    NoiseT = np.einsum('ir,jr,kr->ijk',*U_tilde)       
    NoiseT *= (epsilon / k) 
    
    T_noise = T + NoiseT 
    ###
    # svd_results_V = [svd(covs[i]) for i in range(len(dims))]
    # V_cov = [svd_result[0] for svd_result in svd_results_V]
    # V_cov_k = [V_cov[i][:,:top_sin_val] for i in range(len(dims))] 
    # P_V = [V_cov_k[i] @ V_cov_k[i].T for i in range(len(dims))]

    # svd_results_W = [svd(covs_empirical[i]) for i in range(len(dims))]
    # W_cov_em = [svd_result[0] for svd_result in svd_results_W]
    # W_cov_em_k = [W_cov_em[i][:,:top_sin_val] for i in range(len(dims))] 
    # P_W = [W_cov_em_k[i] @ W_cov_em_k[i].T for i in range(len(dims))]
    
    return T ,T_noise, covs, covs_inv, covs_empirical,covs_pinv_empirical, M_true_inv, M_empirical_pinv
    
     #np.save('/home/maryam/Documents/tensor_decomposition-reorg_cleanup/experiments/results/cov0.npy', covs[0])
    #np.save('/home/maryam/Documents/tensor_decomposition-reorg_cleanup/experiments/results/cov1.npy', covs[1])
    #np.save('/home/maryam/Documents/tensor_decomposition-reorg_cleanup/experiments/results/cov2.npy', covs[2])
    # Generate the tensor T based on the given probabilistic model
    #covs_em = [np.cov(samples.T, rowvar=False) for samples in U]
#random_vectors = [np.random.randn(R, k) for _ in range(N)]  # Shape: [(R, k), (R, k), (R, k)]
    #NoiseT = tenpy.einsum('ir,jr,kr,iR,jR,kR->ijk', *U, *random_vectors)  # Shape: (dims[0], dims[1], dims[2])
####### new model
def generate_tensor_with_noise_new_model(tenpy, dims, R, k, epsilon, alpha, seed=1): 
    N = len(dims)
    U = []
    for i in range(N):
        s = alpha**(-np.linspace(0,R,R))
        A1 = np.random.random((dims[i],R))
        A2 = np.random.random((R,R))

        Q1, RR1 = la.qr(A1)
        Q2, RR2 = la.qr(A2)
        U.append(Q1 @ np.diag(s) @ Q2)
        #U.append(np.random.randn(dims[i], R))  # Random normal matrices of size n[i] x R
    # Generate tensor T
    T = np.einsum('ir,jr,kr->ijk',U[0],U[1],U[2],optimize=True)  
    T /= R 
    ####
    # Compute empirical covariance matrices for each mode
    covs_empirical = [samples @ samples.T for samples in U]
    #covs_empirical = [C/R for C in covs_empirical]
    sample_pinv = [la.pinv(samples) for samples in U]
    covs_pinv_empirical = [samples_pinv.T @ samples_pinv for samples_pinv in sample_pinv]
    # M_empirical_pinv = np.kron(np.kron(covs_pinv_empirical[0],covs_pinv_empirical[1]),covs_pinv_empirical[2])
    # ###
    M_empirical_pinv = covs_pinv_empirical
    # Generate factor matrices for noise
    NoiseT = tenpy.zeros(dims)
    for j in range(k):
        Ux_i = [U[i] @ np.random.randn(R) for i in range(N)] 
        NoiseT += tenpy.einsum('i,j,k->ijk',*Ux_i) 
    NoiseT *= (epsilon / k) 
    T_noise = T + NoiseT
    ###

    # svd_results_W = [svd(covs_empirical[i]) for i in range(len(dims))]
    # W_cov_em = [svd_result[0] for svd_result in svd_results_W]
    # W_cov_em_k = [W_cov_em[i][:,:top_sin_val] for i in range(len(dims))] 
    # P_W = [W_cov_em_k[i] @ W_cov_em_k[i].T for i in range(len(dims))]
    
    return T ,T_noise, covs_empirical, covs_pinv_empirical, M_empirical_pinv
    
    
    


def rand(tenpy, order, s, R, sp_frac=1., seed=1):
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


def neg_rand(tenpy, order, s, R, sp_frac=1., seed=1):
    tenpy.seed(seed * 1001)
    np.random.seed(seed*1001)
    A = []
    for i in range(order):
        if tenpy.name() == 'ctf':
            A.append(tenpy.from_nparray(np.random.uniform(low = -1, high = 1, size=  (s,R))))
        else:
            A.append(np.random.uniform(low = -1, high = 1, size=  (s,R)))
    if sp_frac < 1.:
        O = tenpy.sparse_random([s] * order, 1., 1., sp_frac)
        T = tenpy.TTTP(O, A)
    else:
        T = tenpy.ones([s] * order)
        T = tenpy.TTTP(T, A)
        O = None
    return [T, O]
    
    
def randn(tenpy, order, s, R, sp_frac=1., seed=1):
    tenpy.seed(seed * 1001)
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

#def randn(tenpy, order, s, R, sp_frac=1., seed=1):
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


def rand3(tenpy, s, R, sp_frac=1.):
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


def mm(tenpy, s, R, seed=1):
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


def poisson(s, R):
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


def mom_cons(tenpy, k):
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


def mom_cons_sv(tenpy, k):
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


# def collinearity_tensor(tenpy, s, order, R,
#                              col=[0.2, 0.8],
#                              seed=1):

#     assert(col[0] >= 0. and col[1] <= 1.)
#     assert(s >= R)
#     tenpy.seed(seed * 1001)

#     A = []
#     for i in range(order):
#         #Gamma_L = tenpy.random((s, R))
#         Gamma_L = np.random.randn(s,R)
#         Gamma = tenpy.dot(tenpy.transpose(Gamma_L), Gamma_L)
#         Gamma_min, Gamma_max = Gamma.min(), Gamma.max()
#         Gamma = (Gamma - Gamma_min) / (Gamma_max - Gamma_min) * \
#             (col[1] - col[0]) + col[0]
#         tenpy.fill_diagonal(Gamma, 1.)
#         A_iT = tenpy.cholesky(Gamma)
#         # change size from [R,R] to [s,R]
#         #mat = tenpy.random((s, s))
#         mat = np.random.randn(s,s)
#         [U_mat, sigma_mat, VT_mat] = tenpy.svd(mat)
#         A_iT = tenpy.dot(A_iT, VT_mat[:R, :])

#         A.append(tenpy.transpose(A_iT))
#         col_matrix = tenpy.dot(tenpy.transpose(A[i]), A[i])
#         col_matrix_min, col_matrix_max = col_matrix.min(), (col_matrix - \
#                                                         tenpy.eye(R, R)).max()
#         assert(
#             col_matrix_min -
#             col[0] >= -
#             0.01 and col_matrix_max <= col[1] +
#             0.01)

#     T = tenpy.ones([s] * order)
#     T = tenpy.TTTP(T, A)
#     O = None
#     return [T, O]
def collinearity_tensor(tenpy, s, order, R, k, epsilon,
                             col=[0.2, 0.8],
                             seed=1):

    assert(col[0] >= 0. and col[1] <= 1.)
    assert(s >= R)
    tenpy.seed(seed * 1001)
    dims = [s, s, s]
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
    # Compute empirical covariance matrices for each mode
    covs_empirical = [samples @ samples.T for samples in A]
    #covs_empirical = [C/R for C in covs_empirical]
    sample_pinv = [la.pinv(samples) for samples in A]
    covs_pinv_empirical = [samples_pinv.T @ samples_pinv for samples_pinv in sample_pinv]
    # M_empirical_pinv = np.kron(np.kron(covs_pinv_empirical[0],covs_pinv_empirical[1]),covs_pinv_empirical[2])
    # ###
    M_empirical_pinv = covs_pinv_empirical
    # Generate factor matrices for noise
    NoiseT = tenpy.zeros(dims)
    for j in range(k):
        Ax_i = [A[i] @ np.random.randn(R) for i in range(order)] #Gaussian distribution
        #Ax_i = [A[i] @ np.random.exponential(scale=1.0, size=R) for i in range(order)] #exponential distribution
        #Ax_i = [A[i] @ np.random.beta(a=2.0, b=5.0, size=R) for i in range(order)] #beat disribution
        NoiseT += tenpy.einsum('i,j,k->ijk',*Ax_i) 
    NoiseT *= (epsilon / k) 
    T_noise = T + NoiseT
    ###

    # svd_results_W = [svd(covs_empirical[i]) for i in range(len(dims))]
    # W_cov_em = [svd_result[0] for svd_result in svd_results_W]
    # W_cov_em_k = [W_cov_em[i][:,:top_sin_val] for i in range(len(dims))] 
    # P_W = [W_cov_em_k[i] @ W_cov_em_k[i].T for i in range(len(dims))]
    
    return T , O,  T_noise, covs_empirical, covs_pinv_empirical, M_empirical_pinv
    
