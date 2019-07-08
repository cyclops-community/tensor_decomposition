import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

def name():
    return 'numpy'

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
    T_inds = "".join([chr(ord('a')+i) for i in range(T.ndim)])
    einstr = ""
    A2 = []
    for i in range(len(A)):
        if A[i] is not None:
            einstr += chr(ord('a')+i) + chr(ord('a')+T.ndim) + ','
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

def sparse_random(shape, begin, end, sp_frac):
    tensor = np.random.random(shape)*(end-begin) + begin
    mask = np.random.random(shape)<sp_frac
    tensor = tensor * mask
    return tensor

def vecnorm(T):
    return la.norm(np.ravel(T))

def dot(A,B):
    return np.dot(A,B)

def eigvalh(A):
    return la.eigvalh(A)

def eigvalsh(A):
    return la.eigvalsh(A)

def svd(A,r=None):
    U,s,VT = la.svd(A,full_matrices=False)
    if r is not None:
        U = U[:,:r]
        s = s[:r]
        VT = VT[:r,:]
    return U,s,VT

def svd_rand(A,r=None):
    return svd(A,r)

def cholesky(A):
    return la.cholesky(A)

def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if transp_L:
        A = A.T
    if not from_left:
        B = B.T
        if transp_L:
            llower = lower
        else:
            A = A.T
            llower = not lower
        X = sla.solve_triangular(A, B, llower)
        return X.T
    else:
        return sla.solve_triangular(A, B, lower)

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

def qr(A):
    return la.qr(A)

def reshape(A,shape,order='F'):
    return np.reshape(A,shape,order)

def einsvd(operand, tns, transpose = True, compute_uv = True, full_matrices = True):
    ''' compute SVD of tns

    oprand (str): in form 'src -> tgta, tgtb'
    tns (ndarray): tensor to be decomposed
    transpose (bool): True iff VT(H) is required instead of V
    compute_uv, full_matrices (bool): see numpy.linalg.svd

    REQUIRE: only one contracted index
    '''

    src, _, tgt = operand.replace(' ', '').partition('->')
    tgta, _, tgtb = tgt.partition(',')

    # transpose and reshape to the matrix to be SVD'd
    tgt_idx = set(tgta).union(set(tgtb))
    contract_idx = str(list(tgt_idx.difference(set(src)))[0])
    new_idx = (tgta + tgtb).replace(contract_idx, '')
    trsped = np.einsum(src + '->' + new_idx, tns)

    # do svd
    shape = np.shape(tns)
    letter2size = {}
    for i in range(len(src)):
        letter2size[src[i]] = shape[i]
    col_idx = tgtb.replace(contract_idx, '')
    ncol = 1
    for letter in col_idx:
        ncol *= letter2size[letter]
    mat = np.reshape(trsped, (-1, ncol))
    if not compute_uv:
        s = np.linalg.svd(mat, compute_uv = False)
        return s

    # if u, v are needed
    u, s, vh = np.linalg.svd(mat, full_matrices = full_matrices)

    # reshape u, v into shape (..., contract) and (contract, ...)
    row_idx = tgta.replace(contract_idx, '')
    shapeA = []
    shapeB = [-1]
    for letter in row_idx:
        shapeA.append(letter2size[letter])
    for letter in col_idx:
        shapeB.append(letter2size[letter])
    shapeA.append(-1)
    u = np.reshape(u, shapeA)
    vh = np.reshape(vh, shapeB)

    # transpose u and vh into tgta and tgtb
    preA = tgta.replace(contract_idx, '') + contract_idx
    preB = contract_idx + tgtb.replace(contract_idx, '')
    u = np.einsum(preA + '->' + tgta, u)
    vh = np.einsum(preB + '->' + tgtb, vh)

    # return
    if not transpose:
        vh = np.conj(vh.T)
    return u, s, vh
