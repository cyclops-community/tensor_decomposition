import numpy as np
import sys
import time
from .common_kernels import compute_lin_sys

def solve_sys_lowr_svd(tenpy, G, RHS, r):
    t0 = time.time()
    [U,S,VT] = tenpy.svd(G)
    S = 1./S**.5
    X = tenpy.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    [xU,xS,xVT]=tenpy.svd_rand(X,r)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    0.*xVT.i("ij") << S.i("j") * xVT.i("ij")
    t1 = time.time()
    if tenpy.comm().rank() == 0:
        print("solve system low rank took",t1-t0,"seconds")
    return [xU,tenpy.dot(xVT, VT)]
    #return [X[:,-r:], VT[-r:,:]]

def solve_sys_lowr(tenpy, G, RHS, r):
    L = tenpy.cholesky(G)
    X = tenpy.solve_tri(L, RHS, True, False, True)
    [xU,xS,xVT]=tenpy.svd_rand(X,r)
    #[xU,xS,xVT]=tenpy.svd_rand(X,r,2)
    xVT = tenpy.solve_tri(L, xVT, True, False, False)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    return [xU,xVT]
    #return [X[:,-r:], VT[-r:,:]]

def solve_sys_lowr_sp(tenpy, fG, fRHS, r):
    R = fRHS.shape[1]
    spvec = tenpy.tensor((R),dtype=int)
    spvec.fill_sp_random(1,1,np.float32(r)/R)
    [inds,vals] = spvec.read_all_nnz()
    vals = np.cumsum(vals)
    rr = vals[-1]
    vals[:] -= 1
    VT = tenpy.zeros((rr,R))
    iinds = np.zeros((len(vals),2))
    iinds[:,0] = vals
    iinds[:,1] = inds
    if tenpy.comm().rank() == 0:
        VT.write(iinds,np.ones(len(vals)))
    else:
        VT.write()
    RHS = tenpy.einsum("ik,jk->ij",fRHS,VT)
    G = tenpy.einsum("ik,kl,jl->ij",VT,fG,VT)
    L = tenpy.cholesky(G)
    X = tenpy.solve_tri(L, RHS, True, False, True)
    U = tenpy.solve_tri(L, X, True, False, False)
    return [U,VT]


def build_leaves(tenpy,T,A,B,C):
    TC = tenpy.einsum("ijk,ka->ija",T,C)
    RHS_A = tenpy.einsum("ija,ja->ia",TC,B)
    RHS_B = tenpy.einsum("ija,ia->ja",TC,A)
    RHS_C = tenpy.einsum("ijk,ia,ja->ka",T,A,B)
    return [RHS_A,RHS_B,RHS_C]

def build_leaves_lowr(tenpy,T,A1,A2,B1,B2,C1,C2):
    TA1 = tenpy.einsum("ijk,ir->jkr",T,A1)
    TA1B1 = tenpy.einsum("jkr,jl->krl",TA1,B1)
    A2B2 = tenpy.einsum("ra,la->rla",A2,B2)
    RHS_C = tenpy.einsum("krl,rla->ka",TA1B1,A2B2)
    TA1C1 = tenpy.einsum("jkr,kl->jrl",TA1,C1)
    A2C2 = tenpy.einsum("ra,la->rla",A2,C2)
    RHS_B = tenpy.einsum("jrl,rla->ja",TA1C1,A2C2)
    TC1 = tenpy.einsum("ijk,kl->ijl",T,C1)
    TB1C1 = tenpy.einsum("ijl,jr->irl",TC1,B1)
    B2C2 = tenpy.einsum("ra,la->rla",B2,C2)
    RHS_A = tenpy.einsum("irl,rla->ia",TB1C1,B2C2)

    return [RHS_A,RHS_B,RHS_C]

def update_leaves_A(tenpy,T,A1,A2,B,C):
    TA = tenpy.einsum("ijk,ir,ra->jka",T,A1,A2)
    URHS_B = tenpy.einsum("jka,ka->ja",TA,C)
    URHS_C = tenpy.einsum("jka,ja->ka",TA,B)
    return [URHS_B,URHS_C]


def update_leaves_B(tenpy,T,A,B1,B2,C):
    TB = tenpy.einsum("ijk,jr,ra->ika",T,B1,B2)
    URHS_A = tenpy.einsum("ika,ka->ia",TB,C)
    URHS_C = tenpy.einsum("ika,ia->ka",TB,A)
    return [URHS_A,URHS_C]

def update_leaves_C(tenpy,T,A,B,C1,C2):
    TC = tenpy.einsum("ijk,kr,ra->ija",T,C1,C2)
    URHS_A = tenpy.einsum("ija,ja->ia",TC,B)
    URHS_B = tenpy.einsum("ija,ia->ja",TC,A)
    return [URHS_A,URHS_B]


def update_leaves_sp_A(tenpy,T,A1,A2,B,C):
    TA1 = tenpy.einsum("ijk,ir->jkr",T,A1)
    URHS_B1 = tenpy.einsum("jkr,ka->jra",TA1,C)
    URHS_C1 = tenpy.einsum("jkr,ja->kra",TA1,B)
    URHS_B = tenpy.einsum("jra,ra->ja",URHS_B1,A2)
    URHS_C = tenpy.einsum("kra,ra->ka",URHS_C1,A2)
    return [URHS_B,URHS_C]

def update_leaves_sp_B(tenpy,T,A,B1,B2,C):
    TB1 = tenpy.einsum("ijk,jr->ikr",T,B1)
    URHS_A1 = tenpy.einsum("ikr,ka->ira",TB1,C)
    URHS_C1 = tenpy.einsum("ikr,ia->kra",TB1,A)
    URHS_A = tenpy.einsum("ira,ra->ia",URHS_A1,B2)
    URHS_C = tenpy.einsum("kra,ra->ka",URHS_C1,B2)
    return [URHS_A,URHS_C]

def update_leaves_sp_C(T,A,B,C1,C2):
    TC1 = tenpy.einsum("ijk,kr->ijr",T,C1)
    URHS_A1 = tenpy.einsum("ijr,ja->ira",TC1,B)
    URHS_B1 = tenpy.einsum("ijr,ia->jra",TC1,A)
    URHS_A = tenpy.einsum("ira,ra->ia",URHS_A1,C2)
    URHS_B = tenpy.einsum("jra,ra->ja",URHS_B1,C2)
    return [URHS_A,URHS_B]


def lowr_msdt_step(tenpy,T,A,B,C,RHS_A,RHS_B,RHS_C,r,Regu,ul,uf):
    G = compute_lin_sys(tenpy,B,C,Regu)
    ERHS_A = RHS_A - tenpy.dot(A, G)
    [A1,A2] = globals()[uf](tenpy, G, ERHS_A, r)
    A += tenpy.dot(A1, A2)
    [URHS_B,URHS_C] = globals()[ul+"_A"](tenpy,T,A1,A2,B,C)
    RHS_B += URHS_B
    RHS_C += URHS_C

    G = compute_lin_sys(tenpy,A,C,Regu)
    ERHS_B = RHS_B - tenpy.dot(B, G)
    [B1,B2] = globals()[uf](tenpy, G, ERHS_B, r)
    B += tenpy.dot(B1, B2)
    [URHS_A,URHS_C] = globals()[ul+"_B"](tenpy,T,A,B1,B2,C)
    RHS_A += URHS_A
    RHS_C += URHS_C

    G = compute_lin_sys(tenpy,A,B,Regu)
    ERHS_C = RHS_C - tenpy.dot(C, G)
    [C1,C2] = globals()[uf](tenpy, G, ERHS_C, r)
    C += tenpy.dot(C1, C2)
    [URHS_A,URHS_B] = globals()[ul+"_C"](tenpy,T,A,B,C1,C2)
    RHS_A += URHS_A
    RHS_B += URHS_B

    return [A,B,C,RHS_A,RHS_B,RHS_C]


def lowr_dt_step(tenpy,T,A,B,C,RHS_A,RHS_B,RHS_C,r,Regu,ul,uf,full_rank_factor):
    if not full_rank_factor == "A":
        G = compute_lin_sys(tenpy,B,C,Regu)
        ERHS_A = RHS_A - tenpy.dot(A, G)
        [A1,A2] = globals()[uf](G, ERHS_A, r)
        A += tenpy.dot(A1, A2)
        [URHS_B,URHS_C] = globals()[ul+"_A"](T,A1,A2,B,C)
        RHS_B += URHS_B
        RHS_C += URHS_C
    else: 
        A = solve_sys(tenpy,compute_lin_sys(tenpy,B,C,Regu), RHS_A)
        TA = tenpy.einsum("ijk,ia->jka",T,A)
        RHS_B = tenpy.einsum("jka,ka->ja",TA,C)
        RHS_C = tenpy.einsum("jka,ja->ka",TA,B)

    if not full_rank_factor == "B":
        G = compute_lin_sys(tenpy,A,C,Regu)
        ERHS_B = RHS_B - tenpy.dot(B, G)
        [B1,B2] = globals()[uf](G, ERHS_B, r)
        B += tenpy.dot(B1, B2)
        [URHS_A,URHS_C] = globals()[ul+"_B"](T,A,B1,B2,C)
        RHS_A += URHS_A
        RHS_C += URHS_C
    else:
        B = solve_sys(tenpy,compute_lin_sys(tenpy,A,C,Regu), RHS_B)
        TB = tenpy.einsum("ijk,ja->ika",T,B)
        RHS_A = tenpy.einsum("ika,ka->ia",TB,C)
        RHS_C = tenpy.einsum("ika,ia->ka",TB,A)

    if not full_rank_factor == "C":
        G = compute_lin_sys(tenpy,A,B,Regu)
        ERHS_C = RHS_C - tenpy.dot(C, G)
        [C1,C2] = globals()[uf](G, ERHS_C, r)
        C += tenpy.dot(C1, C2)
        [URHS_A,URHS_B] = globals()[ul+"_C"](T,A,B,C1,C2)
        RHS_A += URHS_A
        RHS_B += URHS_B
    else:
        C = solve_sys(tenpy,compute_lin_sys(tenpy,A,B,Regu), RHS_C)
        TC = tenpy.einsum("ijk,ka->ija",T,C)
        RHS_A = tenpy.einsum("ija,ja->ia",T,B)
        RHS_B = tenpy.einsum("ija,ia->ja",T,A)

    return [A,B,C,RHS_A,RHS_B,RHS_C]

