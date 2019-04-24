import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak

def solve_sys_lowr_svd(G, RHS, r):
    t0 = time.time()
    [U,S,VT] = ctf.svd(G)
    S = 1./S**.5
    X = ctf.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    [xU,xS,xVT]=ctf.svd_rand(X,r)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    0.*xVT.i("ij") << S.i("j") * xVT.i("ij")
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("solve system low rank took",t1-t0,"seconds")
    return [xU,ctf.dot(xVT, VT)]
    #return [X[:,-r:], VT[-r:,:]]

def solve_sys_lowr(G, RHS, r):
    L = ctf.cholesky(G)
    X = ctf.solve_tri(L, RHS, True, False, True)
    [xU,xS,xVT]=ctf.svd_rand(X,r)
    #[xU,xS,xVT]=ctf.svd_rand(X,r,2)
    xVT = ctf.solve_tri(L, xVT, True, False, False)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    return [xU,xVT]
    #return [X[:,-r:], VT[-r:,:]]

def build_leaves(T,A,B,C):
    TC = ctf.einsum("ijk,ka->ija",T,C)
    RHS_A = ctf.einsum("ija,ja->ia",TC,B)
    RHS_B = ctf.einsum("ija,ia->ja",TC,A)
    RHS_C = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    return [RHS_A,RHS_B,RHS_C]

def build_leaves_lowr(T,A1,A2,B1,B2,C1,C2):
    TA1 = ctf.einsum("ijk,ir->jkr",T,A1)
    TA1B1 = ctf.einsum("jkr,jl->krl",TA1,B1)
    A2B2 = ctf.einsum("ra,la->rla",A2,B2)
    RHS_C = ctf.einsum("krl,rla->ka",TA1B1,A2B2)
    TA1C1 = ctf.einsum("jkr,kl->jrl",TA1,C1)
    A2C2 = ctf.einsum("ra,la->rla",A2,C2)
    RHS_B = ctf.einsum("jrl,rla->ja",TA1C1,A2C2)
    TC1 = ctf.einsum("ijk,kl->ijl",T,C1)
    TB1C1 = ctf.einsum("ijl,jr->irl",TC1,B1)
    B2C2 = ctf.einsum("ra,la->rla",B2,C2)
    RHS_A = ctf.einsum("irl,rla->ia",TB1C1,B2C2)

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


def update_leaves_sp_A(T,A1,A2,B,C):
    TA1 = ctf.einsum("ijk,ir->jkr",T,A1)
    URHS_B1 = ctf.einsum("jkr,ka->jra",TA1,C)
    URHS_C1 = ctf.einsum("jkr,ja->kra",TA1,B)
    URHS_B = ctf.einsum("jra,ra->ja",URHS_B1,A2)
    URHS_C = ctf.einsum("kra,ra->ka",URHS_C1,A2)
    return [URHS_B,URHS_C]

def update_leaves_sp_B(T,A,B1,B2,C):
    TB1 = ctf.einsum("ijk,jr->ikr",T,B1)
    URHS_A1 = ctf.einsum("ikr,ka->ira",TB1,C)
    URHS_C1 = ctf.einsum("ikr,ia->kra",TB1,A)
    URHS_A = ctf.einsum("ira,ra->ia",URHS_A1,B2)
    URHS_C = ctf.einsum("kra,ra->ka",URHS_C1,B2)
    return [URHS_A,URHS_C]

def update_leaves_sp_C(T,A,B,C1,C2):
    TC1 = ctf.einsum("ijk,kr->ijr",T,C1)
    URHS_A1 = ctf.einsum("ijr,ja->ira",TC1,B)
    URHS_B1 = ctf.einsum("ijr,ia->jra",TC1,A)
    URHS_A = ctf.einsum("ira,ra->ia",URHS_A1,C2)
    URHS_B = ctf.einsum("jra,ra->ja",URHS_B1,C2)
    return [URHS_A,URHS_B]


def lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,Regu,ul="update_leaves"):
    G = cak.compute_lin_sys(B,C,Regu)
    ERHS_A = RHS_A - ctf.dot(A, G)
    [A1,A2] = solve_sys_lowr(G, ERHS_A, r)
    A += ctf.dot(A1, A2)
    [URHS_B,URHS_C] = globals()[ul+"_A"](T,A1,A2,B,C)
    RHS_B += URHS_B
    RHS_C += URHS_C

    G = cak.compute_lin_sys(A,C,Regu)
    ERHS_B = RHS_B - ctf.dot(B, G)
    [B1,B2] = solve_sys_lowr(G, ERHS_B, r)
    B += ctf.dot(B1, B2)
    [URHS_A,URHS_C] = globals()[ul+"_B"](T,A,B1,B2,C)
    RHS_A += URHS_A
    RHS_C += URHS_C

    G = cak.compute_lin_sys(A,B,Regu)
    ERHS_C = RHS_C - ctf.dot(C, G)
    [C1,C2] = solve_sys_lowr(G, ERHS_C, r)
    C += ctf.dot(C1, C2)
    [URHS_A,URHS_B] = globals()[ul+"_C"](T,A,B,C1,C2)
    RHS_A += URHS_A
    RHS_B += URHS_B

    return [A,B,C,RHS_A,RHS_B,RHS_C]


