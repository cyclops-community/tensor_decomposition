import ctf
import numpy as np

def name():
    return 'ctf'

def save_tensor_to_file(T, filename):
    np.save(filename, T.to_nparray())

def load_tensor_from_file(filename):
    try:
        T = np.load(filename)
        print('Loaded tensor from file ', filename)
    except FileNotFoundError:
        raise FileNotFoundError('No tensor exist on: ', filename)
    return ctf.from_nparray(T)

def from_nparray(arr):
    return ctf.from_nparray(arr)

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
    
def list_vecnormsq(list_A):
    l = [i**2 for i in list_A]
    s = 0
    for i in range(len(l)):
        s+=ctf.sum(l[i])
    return s
    
def list_vecnorm(list_A):
    l = [i**2 for i in list_A]
    s = 0
    for i in range(len(l)):
        s+=ctf.sum(l[i])
        
    return s**0.5
    
    
def mult_lists(list_A,list_B):
    l=[A*B for (A,B) in zip(list_A,list_B)]
    s= 0
    for i in range(len(l)):
        s+=ctf.sum(l[i])
        
    return s
    

def norm(v):
    return v.norm2() 

def dot(A,B):
    return ctf.dot(A,B)

def svd(A,r=None):
    return ctf.svd(A,r)

def svd_rand(A,r):
    return ctf.svd_rand(A,r)

def cholesky(A):
    return ctf.cholesky(A)

def solve_tri(A, B, lower=True, from_left=False, transp_L=False):
    return ctf.solve_tri(A, B, lower, from_left, transp_L)

def einsum(string, *args):
    if "..." in string:
        left = string.split(",")
        left[-1], right = left[-1].split("->")
        symbols = "".join(list(set([chr(i) for i in range(48, 127)]) - set(string.replace(".", "").replace(",", "").replace("->", ""))))
        symbol_idx = 0
        for i, (s, tsr) in enumerate(zip(left, args)):
            num_missing = tsr.ndim - len(s.replace("...", ""))
            left[i] = s.replace("...", symbols[symbol_idx:symbol_idx+num_missing])
            symbol_idx += num_missing
        right = right.replace("...", symbols[:symbol_idx])
        string = ",".join(left) + "->" + right

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
    
def list_add(list_A,list_B):
    return [A+B for (A,B) in zip(list_A,list_B)]
    
def scalar_mul(sclr,list_A):
    return [sclr*A for A in list_A]

def speye(*args):
    return ctf.speye(*args)

def eye(*args):
    return ctf.eye(*args)

def transpose(A):
    return ctf.transpose(A)

def argmax(A, axis=0):
    return abs(A).to_nparray().argmax(axis=axis)

def asarray(T):
    return ctf.astensor(T)
    

def reshape(A,shape,order='F'):
    return ctf.reshape(A,shape,order)

def einsvd(einstr, A, r=None, transpose=True, compute_uv=True, full_matrices=True, mult_sv=False):
    str_a, str_uv = einstr.split("->")
    str_u, str_v = str_uv.split(",")
    U, S, Vh = A.i(str_a).svd(str_u, str_v, rank=r)
    if not compute_uv:
        return S
    if mult_sv:
        char_i = list(set(str_v) - set(str_a))[0]
        char_s = list(set(str_a) - set(str_v))[0]
        Vh = ctf.einsum(char_s + char_i + "," + str_v + "->" + str_v.replace(char_i, char_s), ctf.diag(S), Vh)
    if not transpose:
        Vh = Vh.T
    return U, S, Vh

def squeeze(A):
    return A.reshape([s for s in A.shape if s != 1])
