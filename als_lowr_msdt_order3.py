import numpy as np
import numpy.linalg as la
import ctf

def compute_lin_sys(X, Y):
    return (X.T() @ X) * (Y.T() @ Y)

def solve_sys(G, RHS):
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = RHS @ U
    0.*X.i("ij") << S.i("j") * X.i("ij")
    return X @ VT

def naive_ALS_step(T,A,B,C):
    RHS = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    A = solve_sys(compute_lin_sys(B,C), RHS)
    RHS = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    B = solve_sys(compute_lin_sys(A,C), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(compute_lin_sys(A,B), RHS)
    return [A,B,C]

def get_residual(T,A,B,C):
    return ctf.vecnorm(T-ctf.einsum("ia,ja,ka->ijk",A,B,C))

def init_rand(s,R):
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    T = ctf.einsum("ia,ja,ka->ijk",A,B,C)
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    return [A,B,C,T]

def test_rand_naive(s,R):
    [A,B,C,T] = init_rand(s,R)
    for i in range(10):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C] = naive_ALS_step(T,A,B,C)


s = 10
R = 4
r = 2

test_rand_naive(s,R)
