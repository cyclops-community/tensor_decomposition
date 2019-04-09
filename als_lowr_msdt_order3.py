import numpy as np
import numpy.linalg as la
import ctf
import sys
from ctf import random
import time

def compute_lin_sys(X, Y):
    return ctf.dot(X.T(), X) * ctf.dot(Y.T(), Y)

def solve_sys_svd(G, RHS):
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = ctf.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    return ctf.dot(X, VT)

def solve_sys(G, RHS):
    L = ctf.cholesky(G)
    X = ctf.solve_tri(L, RHS, True, False, True)
    X = ctf.solve_tri(L, X, True, False, False)
    return X

def get_residual_sp(O,T,A,B,C):
    t0 = time.time()
    nrm1 = ctf.vecnorm(ctf.TTTP(O,[A,B,C]))**2
    nrm2 = ctf.sum(ctf.einsum("ia,ib,ja,jb,ka,kb->ab",A,A,B,B,C,C))
    nrm3 = ctf.vecnorm(T - ctf.TTTP(O,[A,B,C]))**2
    nrm = (nrm3+nrm2-nrm1)**.5
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Sparse residual computation took",t1-t0,"seconds")
    return nrm
   

def get_residual(T,A,B,C):
    t0 = time.time()
    nrm = ctf.vecnorm(T-ctf.einsum("ia,ja,ka->ijk",A,B,C))
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Residual computation took",t1-t0,"seconds")
    return nrm


def naive_ALS_step(T,A,B,C):
    RHS = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    A = solve_sys(compute_lin_sys(B,C), RHS)
    RHS = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    B = solve_sys(compute_lin_sys(A,C), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(compute_lin_sys(A,B), RHS)
    return [A,B,C]

def dt_ALS_step(T,A,B,C):
    TC = ctf.einsum("ijk,ka->ija",T,C)
    RHS = ctf.einsum("ija,ja->ia",TC,B)
    A = solve_sys(compute_lin_sys(B,C), RHS)
    RHS = ctf.einsum("ija,ia->ja",TC,A)
    B = solve_sys(compute_lin_sys(A,C), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(compute_lin_sys(A,B), RHS)
    return [A,B,C]

def build_leaves(T,A,B,C):
    TC = ctf.einsum("ijk,ka->ija",T,C)
    RHS_A = ctf.einsum("ija,ja->ia",TC,B)
    RHS_B = ctf.einsum("ija,ia->ja",TC,A)
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

#def solve_sys_lowr_post_fac(G, RHS, r):
#    [U,S,VT] = ctf.svd(G)
#    S = 1./S
#    X = ctf.dot(RHS, U)
#    0.*X.i("ij") << S.i("j") * X.i("ij")
#    [U,S,VT]=ctf.dot(ctf.svd(X, ) VT)
#    0.*VT.i("ij") << S.i("i") * VT.i("ij")
#    return [U[:,:r],VT[:r,:]]
#    #return [X[:,-r:], VT[-r:,:]]


def solve_sys_lowr_svd(G, RHS, r):
    [U,S,VT] = ctf.svd(G)
    S = 1./S**.5
    X = ctf.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    [xU,xS,xVT]=ctf.svd(X,r)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    0.*xVT.i("ij") << S.i("j") * xVT.i("ij")
    return [xU,ctf.dot(xVT, VT)]
    #return [X[:,-r:], VT[-r:,:]]

def solve_sys_lowr(G, RHS, r):
    L = ctf.cholesky(G)
    X = ctf.solve_tri(L, RHS, True, False, True)
    [xU,xS,xVT]=ctf.svd(X,r)
    xVT = ctf.solve_tri(L, xVT, True, False, False)
    0.*xVT.i("ij") << xS.i("i") * xVT.i("ij")
    return [xU,xVT]
    #return [X[:,-r:], VT[-r:,:]]



def lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,ul="update_leaves"):
    G = compute_lin_sys(B,C)
    ERHS_A = RHS_A - ctf.dot(A, G)
    [A1,A2] = solve_sys_lowr(G, ERHS_A, r)
    A += ctf.dot(A1, A2)
    [URHS_B,URHS_C] = globals()[ul+"_A"](T,A1,A2,B,C)
    RHS_B += URHS_B
    RHS_C += URHS_C

    G = compute_lin_sys(A,C)
    ERHS_B = RHS_B - ctf.dot(B, G)
    [B1,B2] = solve_sys_lowr(G, ERHS_B, r)
    B += ctf.dot(B1, B2)
    [URHS_A,URHS_C] = globals()[ul+"_B"](T,A,B1,B2,C)
    RHS_A += URHS_A
    RHS_C += URHS_C

    G = compute_lin_sys(A,B)
    ERHS_C = RHS_C - ctf.dot(C, G)
    [C1,C2] = solve_sys_lowr(G, ERHS_C, r)
    C += ctf.dot(C1, C2)
    [URHS_A,URHS_B] = globals()[ul+"_C"](T,A,B,C1,C2)
    RHS_A += URHS_A
    RHS_B += URHS_B

    return [A,B,C,RHS_A,RHS_B,RHS_C]

def init_rand(s,R,sp_frac=1.):
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    if sp_frac<1.:
        O = ctf.tensor((s,s,s),sp=True)
        O.fill_sp_random(1.,1.,sp_frac)
        T = ctf.TTTP(O,[A,B,C])
    else:
        T = ctf.einsum("ia,ja,ka->ijk",A,B,C)
        O = None
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    return [A,B,C,T,O]

def test_rand_naive(s,R,num_iter,sp_frac,sp_res):
    [A,B,C,T,O] = init_rand(s,R,sp_frac)
    time_all = 0.
    for i in range(num_iter):
        if sp_res:
            res = get_residual_sp(O,T,A,B,C)
        else:
            res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        [A,B,C] = dt_ALS_step(T,A,B,C)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    if ctf.comm().rank() == 0:
        print("Naive method took",time_all,"seconds overall")

def test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul=False,sp_res=False):
    [A,B,C,T,O] = init_rand(s,R,sp_frac)
    time_init = 0.
    for i in range(num_lowr_init_iter):
        if sp_res:
            res = get_residual_sp(O,T,A,B,C)
        else:
            res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        [A,B,C] = dt_ALS_step(T,A,B,C)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Full-rank sweep took", t1-t0,"seconds")
        time_init += t1-t0
    time_lowr = time.time()
    [RHS_A,RHS_B,RHS_C] = build_leaves(T,A,B,C)
    time_lowr -= time.time()
    for i in range(num_iter-num_lowr_init_iter):
        if sp_res:
            res = get_residual_sp(O,T,A,B,C)
        else:
            res = get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        if sp_ul:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,"update_leaves_sp")
        else:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Low-rank sweep took", t1-t0,"seconds")
        time_lowr += t1-t0
    if ctf.comm().rank() == 0:
        print("Low rank method (sparse update leaves =",sp_ul,") took",time_init,"for initial full rank steps",time_lowr,"for low rank steps and",time_init+time_lowr,"seconds overall")


if __name__ == "__main__":
    w = ctf.comm()
    s = 40
    R = 10
    r = 10
    num_iter = 10
    num_lowr_init_iter = 2
    sp_frac = 1.
    sp_ul = 0
    sp_res = 0
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
        sp_frac  = np.float64(sys.argv[6])
    if len(sys.argv) >= 8:
        sp_ul  = int(sys.argv[7])
    if len(sys.argv) >= 9:
        sp_res  = int(sys.argv[8])
    if len(sys.argv) >= 10:
        run_naive = int(sys.argv[9])
    if len(sys.argv) >= 11:
        run_lowr = int(sys.argv[10])

    if ctf.comm().rank() == 0:
        print("Arguments to exe are (s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul,sp_res,run_naive,run_lowr), default is (",40,10,10,10,2,1.,1,0,1,1,")provided", sys.argv)
    if run_naive:
        if ctf.comm().rank() == 0:
            print("Testing naive version, printing residual before every ALS sweep")
        test_rand_naive(s,R,num_iter,sp_frac,sp_res)
    if run_lowr:
        if ctf.comm().rank() == 0:
            print("Testing low rank version, printing residual before every ALS sweep")
        test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul,sp_res)
