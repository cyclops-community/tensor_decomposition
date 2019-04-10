import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak
import lowr_ALS3 as lALS
import standard_ALS3 as sALS
import synthetic_tensors as stsrs

def test_rand_naive(s,R,num_iter,sp_frac,sp_res,mm_test=False):
    if mm_test == True:
        [A,B,C,T,O] = stsrs.init_mm(s,R)
    else:
        [A,B,C,T,O] = stsrs.init_rand(s,R,sp_frac)
    time_all = 0.
    for i in range(num_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        [A,B,C] = sALS.dt_ALS_step(T,A,B,C)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    if ctf.comm().rank() == 0:
        print("Naive method took",time_all,"seconds overall")

def test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul=False,sp_res=False,mm_test=False):
    if mm_test == True:
        [A,B,C,T,O] = stsrs.init_mm(s,R)
    else:
        [A,B,C,T,O] = stsrs.init_rand(s,R,sp_frac)
    time_init = 0.
    for i in range(num_lowr_init_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        [A,B,C] = sALS.dt_ALS_step(T,A,B,C)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Full-rank sweep took", t1-t0,"seconds, iteration ",i)
        time_init += t1-t0
        if ctf.comm().rank() == 0:
            print ("Total time is ", time_init)
    time_lowr = time.time()
    if num_lowr_init_iter == 0:
        if ctf.comm().rank() == 0:
            print("Initializing leaves from low rank factor matrices")
        A1 = ctf.random.random((s,r))
        B1 = ctf.random.random((s,r))
        C1 = ctf.random.random((s,r))
        A2 = ctf.random.random((r,R))
        B2 = ctf.random.random((r,R))
        C2 = ctf.random.random((r,R))
        [RHS_A,RHS_B,RHS_C] = lALS.build_leaves_lowr(T,A1,A2,B1,B2,C1,C2)
        A = ctf.dot(A1,A2)
        B = ctf.dot(B1,B2)
        C = ctf.dot(C1,C2)
        if ctf.comm().rank() == 0:
            print("Done initializing leaves from low rank factor matrices")
    else:
        [RHS_A,RHS_B,RHS_C] = lALS.build_leaves(T,A,B,C)
    time_lowr = time.time() - time_lowr
    for i in range(num_iter-num_lowr_init_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
        t0 = time.time()
        if sp_ul:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lALS.lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,"update_leaves_sp")
        else:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lALS.lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Low-rank sweep took", t1-t0,"seconds, Iteration",i)
        time_lowr += t1-t0
        if ctf.comm().rank() == 0:
            print("Total time is ", time_lowr)
    if ctf.comm().rank() == 0:
        print("Low rank method (sparse update leaves =",sp_ul,") took",time_init,"for initial full rank steps",time_lowr,"for low rank steps and",time_init+time_lowr,"seconds overall")


if __name__ == "__main__":
    w = ctf.comm()
    s = 64
    R = 10
    r = 10
    num_iter = 10
    num_lowr_init_iter = 2
    sp_frac = 1.
    sp_ul = 0
    sp_res = 0
    run_naive = 1
    run_lowr = 1
    run_lowr = 1
    mm_test = 0
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
    if len(sys.argv) >= 12:
        mm_test = int(sys.argv[11])

    if ctf.comm().rank() == 0:
        #print("Arguments to exe are (s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul,sp_res,run_naive,run_lowr,mm_test), default is (",40,10,10,10,2,1.,1,0,1,1,1,")provided", sys.argv)
        print("s =",s)
        print("R =",R)
        print("r =",r)
        print("num_iter =",num_iter)
        print("num_lowr_init_iter =",num_lowr_init_iter)
        print("sp_frac =",sp_frac)
        print("sp_ul (mem-preserving ordering of low-rank sparse contractions) =",sp_ul)
        print("sp_res (TTTP-based sparse residual calculation) =",sp_res)
        print("run_naive =",run_naive)
        print("run_lowr =",run_lowr)
        print("mm_test (decompose matrix multiplication tensor as opposed to random) =",mm_test)
    if run_naive:
        if ctf.comm().rank() == 0:
            print("Testing naive version, printing residual before every ALS sweep")
        test_rand_naive(s,R,num_iter,sp_frac,sp_res,mm_test)
    if run_lowr:
        if ctf.comm().rank() == 0:
            print("Testing low rank version, printing residual before every ALS sweep")
        test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul,sp_res,mm_test)
