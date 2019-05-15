import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

def TTTP(T, A):
    T_inds = "".join([chr(ord('a')+i) for i in range(T.ndim)])
    einstr = ""
    A2 = []
    for i in range(len(A)):
        if A[i] is not None:
            einstr += chr(ord('a')+i) + chr(ord('a')+T.ndim) + ','
            A2.append(A[i])
    einstr += T_inds + "->" + T_inds
    return np.einsum(einstr, *A2, T)

def is_master_proc():
    return True

def printf(*string):
    print(string)

def tensor(shape, sp, *args2):
    return np.ndarray(shape, args2)

def fill_sp_random(tensor, begin, end, sp_frac):
    tensor = np.asarray(np.random.random(tensor.shape), dtype=tensor.dtype)*(end-begin) + begin
    mask = np.random.random(tensor.shape)<sp_frac
    tensor = tensor * mask

def vecnorm(T):
    return la.norm(np.ravel(T))

def dot(A,B):
    return np.dot(A,B)

def svd(A):
    return la.svd(A)

def cholesky(A):
    return la.cholesky(A)

def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if transp_L:
        A = A.T
    if not from_left:
        B = B.T
        A = A.T
        X = sla.solve_triangular(A, B, lower)
        return X.T
    else:
        return sla.solve_triangular(A, B, lower)

def einsum(string, *args, out=None):
    if out is None:
        return np.einsum(string, *args)
    else:
        out = np.einsum(string, *args)
        return out

def ones(shape):
    return np.ones(shape)

def sum(A, axes=None):
    return np.sum(A, axes)

def random(shape):
    return np.random.random(shape)

def speye(*args):
    return np.eye(*args)

def eye(*args):
    return np.eye(*args)

def transpose(A):
    return A.T
