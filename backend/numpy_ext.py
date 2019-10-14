import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


def name():
    return 'numpy'


def fill_diagonal(matrix, value):
    return np.fill_diagonal(matrix, value)


def diag(v):
    return np.diag(v)


def save_tensor_to_file(T, filename):
    np.save(filename, T)


def load_tensor_from_file(filename):
    try:
        T = np.load(filename)
        print('Loaded tensor from file ', filename)
    except FileNotFoundError:
        raise FileNotFoundError('No tensor exist on: ', filename)
    return T


def TTTP(T, A):
    """ Tensor Times Tensor Product
    """
    T_inds = "".join([chr(ord('a') + i) for i in range(T.ndim)])
    einstr = ""
    A2 = []
    for i in range(len(A)):
        if A[i] is not None:
            einstr += chr(ord('a') + i) + chr(ord('a') + T.ndim) + ','
            A2.append(A[i])
    einstr += T_inds + "->" + T_inds
    A2.append(T)
    return np.einsum(einstr, *A2)


def is_master_proc():
    return True


def printf(*string):
    print(string)


def tensor(shape, sp, *args2):
    return np.ndarray(shape, *args2)


def list_add(list_A, list_B):
    return [A + B for (A, B) in zip(list_A, list_B)]


def scalar_mul(sclr, list_A):
    return [sclr * A for A in list_A]


def mult_lists(list_A, list_B):
    l = [A * B for (A, B) in zip(list_A, list_B)]
    s = 0
    for i in range(len(l)):
        s += np.sum(l[i])

    return s


def list_vecnormsq(list_A):
    """ Vector Norm square of the list 
    """
    l = [i**2 for i in list_A]
    return np.sum(l)


def list_vecnorm(list_A):
    l = [i**2 for i in list_A]
    s = 0
    for i in range(len(l)):
        s += np.sum(l[i])

    return s**0.5


def sparse_random(shape, begin, end, sp_frac):
    tensor = np.random.random(shape) * (end - begin) + begin
    mask = np.random.random(shape) < sp_frac
    tensor = tensor * mask
    return tensor


def vecnorm(T):
    return la.norm(np.ravel(T))


def norm(v):
    return la.norm(v)


def dot(A, B):
    return np.dot(A, B)


def svd(A, r=None):
    U, s, VT = la.svd(A, full_matrices=False)
    if r is not None:
        U = U[:, :r]
        s = s[:r]
        VT = VT[:r, :]
    return U, s, VT


def svd_rand(A, r=None):
    return svd(A, r)


def cholesky(A):
    return la.cholesky(A)


def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if not from_left:
        B = B.T
        A = A.T
        llower = not lower
        X = sla.solve_triangular(A, B, trans=transp_L, lower=llower)
        return X.T
    else:
        return sla.solve_triangular(A, B, trans=transp_L, lower=lower)


def einsum(string, *args):
    out = np.einsum(string, *args)
    return out


def ones(shape):
    return np.ones(shape)


def zeros(shape):
    return np.zeros(shape)


def sum(A, axes=None):
    return np.sum(A, axes)


def random(shape):
    return np.random.random(shape)


def seed(seed):
    return np.random.seed(seed)


def speye(*args):
    return np.eye(*args)


def eye(*args):
    return np.eye(*args)


def transpose(A):
    return A.T


def argmax(A, axis=0):
    return abs(A).argmax(axis=axis)


def asarray(T):
    return np.array(T)


def reshape(A, shape, order='F'):
    return np.reshape(A, shape, order)


def squeeze(A):
    return A.squeeze()
