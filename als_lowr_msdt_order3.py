import numpy as np
import numpy.linalg as la
import ctf
import sys
from ctf import random

def compute_lin_sys(X, Y):
    return (X.T() @ X) * (Y.T() @ Y)

def solve_sys(G, RHS):
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = RHS @ U
    0.*X.i("ij") << S.i("j") * X.i("ij")
    return X @ VT

def get_residual(T,A,B,C):
    return ctf.vecnorm(T-ctf.einsum("ia,ja,ka->ijk",A,B,C))

def naive_ALS_step(T,A,B,C):
    RHS = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    A = solve_sys(compute_lin_sys(B,C), RHS)
    RHS = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    B = solve_sys(compute_lin_sys(A,C), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(compute_lin_sys(A,B), RHS)
    return [A,B,C]

def build_leaves(T,A,B,C):
    RHS_A = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    RHS_B = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    RHS_C = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    return [RHS_A,RHS_B,RHS_C]

def update_leaves_A(T,A1,A2,B,C):
    TA1 = ctf.einsum("ijk,ir->jkr",T,A1)
    URHS_B1 = ctf.einsum("jkr,ka->jra",TA1,C)
    URHS_C1 = ctf.einsum("jkr,ja->kra",TA1,B)
    URHS_B = ctf.einsum("jra,ra->ja",URHS_B1,A2)
    URHS_C = ctf.einsum("kra,ra->ka",URHS_C1,A2)
    return [URHS_B,URHS_C]

def update_leaves_B(T,A,B1,B2,C):
    TB = ctf.einsum("ijk,jr,ra->ika",T,B1,B2)
    URHS_A = ctf.einsum("ika,ka->ia",TB,C)
    URHS_C = ctf.einsum("ika,ia->ka",TB,A)
    return [URHS_A,URHS_C]

def update_leaves_C(T,A,B,C1,C2):
    TC = ctf.einsum("ijk,kr,ra->ija",T,C1,C2)
    URHS_A = ctf.einsum("ija,ja->ia",TC,B)
    URHS_B = ctf.einsum("ija,ia->ja",TC,A)
    return [URHS_A,URHS_B]

def solve_sys_lowr_post_fac(G, RHS, r):
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = RHS @ U
    0.*X.i("ij") << S.i("j") * X.i("ij")
    [U,S,VT]=ctf.svd(X@ VT)
    0.*VT.i("ij") << S.i("i") * VT.i("ij")
    return [U[:,:r],VT[:r,:]]
    #return [X[:,-r:], VT[-r:,:]]


def solve_sys_lowr(G, RHS, r):
    [U,S,VT] = ctf.svd(G)
    S = 1./S**.5
    X = RHS @ U
    0.*X.i("ij") << S.i("j") * X.i("ij")
    [xU,xS,xVT]=ctf.svd(X,r)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    0.*xVT.i("ij") << S.i("j") * xVT.i("ij")
    return [xU,xVT@VT]
    #return [X[:,-r:], VT[-r:,:]]


def lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r):
    G = compute_lin_sys(B,C)
    ERHS_A = RHS_A - A @ G
    [A1,A2] = solve_sys_lowr(G, ERHS_A, r)
    A += A1 @ A2
    [URHS_B,URHS_C] = update_leaves_A(T,A1,A2,B,C)
    RHS_B += URHS_B
    RHS_C += URHS_C

    G = compute_lin_sys(A,C)
    ERHS_B = RHS_B - B @ G
    [B1,B2] = solve_sys_lowr(G, ERHS_B, r)
    B += B1 @ B2
    [URHS_A,URHS_C] = update_leaves_B(T,A,B1,B2,C)
    RHS_A += URHS_A
    RHS_C += URHS_C

    G = compute_lin_sys(A,B)
    ERHS_C = RHS_C - C @ G
    [C1,C2] = solve_sys_lowr(G, ERHS_C, r)
    C += C1 @ C2
    [URHS_A,URHS_B] = update_leaves_C(T,A,B,C1,C2)
    RHS_A += URHS_A
    RHS_B += URHS_B

    return [A,B,C,RHS_A,RHS_B,RHS_C]

def init_rand(s,R):
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    T = ctf.einsum("ia,ja,ka->ijk",A,B,C)
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    return [A,B,C,T]

def test_rand_naive(s,R,num_iter):
    [A,B,C,T] = init_rand(s,R)
    for i in range(num_iter):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C] = naive_ALS_step(T,A,B,C)

def test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter):
    [A,B,C,T] = init_rand(s,R)
    for i in range(num_lowr_init_iter):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C] = naive_ALS_step(T,A,B,C)
    [RHS_A,RHS_B,RHS_C] = build_leaves(T,A,B,C)
    for i in range(num_iter-num_lowr_init_iter):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r)


if __name__ == "__main__":
    s = 40
    R = 10
    r = 10
    num_iter = 10
    num_lowr_init_iter = 2
    run_naive = 1
    run_lowr = 1
    if len(sys.argv) >= 4:
        s = int(sys.argv[1])
        R = int(sys.argv[2])
        r = int(sys.argv[3])
    if len(sys.argv) >= 5:
        num_iter = int(sys.argv[4])
    if len(sys.argv) >= 6:
        num_lowr_init_iter = int(sys.argv[5])
    if len(sys.argv) >= 7:
        run_naive = int(sys.argv[6])
    if len(sys.argv) >= 8:
        run_lowr = int(sys.argv[6])

    if ctf.comm().rank() == 0:
        print("Arguments to exe are (s,R,r,num_iter,num_lowr_init_iter,run_naive,run_lowr), default is (",40,10,10,10,2,1,1,")provided", sys.argv)
    if run_naive:
        if ctf.comm().rank() == 0:
            print("Testing naive version, printing residual before every ALS sweep")
        test_rand_naive(s,R,num_iter)
    if run_lowr:
        if ctf.comm().rank() == 0:
            print("Testing low rank version, printing residual before every ALS sweep")
        test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter)
