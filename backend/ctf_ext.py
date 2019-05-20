import ctf

def TTTP(T, A):
    return ctf.TTTP(T, A)

def is_master_proc():
    if ctf.comm().rank() == 0:
        return True
    else:
        return False

def printf(*string):
    if ctf.comm().rank() == 0:
        print(string)

def tensor(shape, sp, *args):
    return ctf.tensor(shape, sp, *args)
        
def sparse_random(shape, begin, end, sp_frac):
    tensor = ctf.tensor(shape, sp=True)
    tensor.fill_sp_random(begin, end, sp_frac)
    return tensor

def vecnorm(T):
    return ctf.vecnorm(T)

def dot(A,B):
    return ctf.dot(A,B)

def svd(A):
    return ctf.svd(A)

def svd_rand(A,r):
    return ctf.svd_rand(A,r)

def cholesky(A):
    return ctf.cholesky(A)

def solve_tri(A, B, lower=True, from_left=False, transp_L=False):
    return ctf.solve_tri(A, B, lower, from_left, transp_L)

def einsum(string, *args):
    return ctf.einsum(string, *args)

def ones(shape):
    return ctf.ones(shape)

def zeros(shape):
    return ctf.zeros(shape)

def sum(A, axes=None):
    return ctf.sum(A, axes)

def random(shape):
    return ctf.random.random(shape)

def seed(seed):
    return ctf.random.seed(seed)

def speye(*args):
    return ctf.speye(*args)

def eye(*args):
    return ctf.eye(*args)

def transpose(A):
    return ctf.transpose(A)

def argmax(A, axis=0):
    return abs(A).to_nparray().argmax(axis=axis)
