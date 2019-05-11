import numpy as np
import ctf
from ctf import random
import sys
import time

def compute_lin_sysN(A,i,Regu):
    S = None
    for j in range(len(A)):
        if j != i:
            if S is None:
                S =  ctf.dot(A[j].T(), A[j])
            else:
                S *= ctf.dot(A[j].T(), A[j])
    S += Regu
    return S

def compute_lin_sys(X, Y, Regu):
    return ctf.dot(X.T(), X) * ctf.dot(Y.T(), Y) + Regu

def solve_sys_svd(G, RHS):
    t0 = time.time()
    [U,S,VT] = ctf.svd(G)
    S = 1./S
    X = ctf.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Solving linear system took ",t1-t0,"seconds")
    return ctf.dot(X, VT)

def solve_sys(G, RHS):
    L = ctf.cholesky(G)
    X = ctf.solve_tri(L, RHS, True, False, True)
    X = ctf.solve_tri(L, X, True, False, False)
    return X

def get_residual3(T,A,B,C):
    t0 = time.time()
    nrm = ctf.vecnorm(T-ctf.einsum("ia,ja,ka->ijk",A,B,C))
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Residual computation took",t1-t0,"seconds")
    return nrm

def get_residual_sp3(O,T,A,B,C):
    t0 = time.time()
    K = ctf.TTTP(O,[A,B,C])
    nrm1 = ctf.vecnorm(K)**2
    nrm2 = ctf.sum(ctf.dot(A.T(),A)*ctf.dot(B.T(),B)*ctf.dot(C.T(),C))
    #ctf.einsum("ia,ib,ja,jb,ka,kb->ab",A,A,B,B,C,C))
    diff = T - K
    nrm3 = ctf.vecnorm(diff)**2
    nrm = (nrm3+nrm2-nrm1)**.5
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Sparse residual computation took",t1-t0,"seconds")
    return nrm

def get_residual(T,A):
    t0 = time.time()
    V = ctf.ones(T.shape)
    nrm = ctf.vecnorm(T-ctf.TTTP(V,A))
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Residual computation took",t1-t0,"seconds")
    return nrm

def get_residual_sp(O,T,A):
    t0 = time.time()
    K = ctf.TTTP(O,A)
    nrm1 = ctf.vecnorm(K)**2
    B = ctf.dot(A[0].T(),A[0])
    for i in range(1,len(A)):
      B *= ctf.dot(A[i].T(),A[i])
    nrm2 = ctf.sum(B)
    diff = T - K
    nrm3 = ctf.vecnorm(diff)**2
    nrm = (nrm3+nrm2-nrm1)**.5
    t1 = time.time()
    if ctf.comm().rank() == 0:
        print("Sparse residual computation took",t1-t0,"seconds")
    return nrm
