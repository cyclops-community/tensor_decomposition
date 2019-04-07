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
    TA = ctf.einsum("ijk,ir,ra->jka",T,A1,A2)
    URHS_B = ctf.einsum("jka,ka->ja",TA,C)
    URHS_C = ctf.einsum("jka,ja->ka",TA,B)
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

def solve_sys_lowr(G, RHS, r):
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = RHS @ U
    0.*X.i("ij") << S.i("j") * X.i("ij")
    return [X[:,:r], VT[:r,:]]


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

def test_rand_naive(s,R):
    [A,B,C,T] = init_rand(s,R)
    for i in range(10):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C] = naive_ALS_step(T,A,B,C)

def test_rand_lowr(s,R,r):
    [A,B,C,T] = init_rand(s,R)
    for i in range(2):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C] = naive_ALS_step(T,A,B,C)
    [RHS_A,RHS_B,RHS_C] = build_leaves(T,A,B,C)
    for i in range(8):
        res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r)


s = 40
R = 10
r = 10

test_rand_naive(s,R)
test_rand_lowr(s,R,r)
