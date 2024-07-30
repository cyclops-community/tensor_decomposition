import numpy as np
import sys
import time

def compute_lin_sysN(tenpy, A,i,Regu):
    S = None
    for j in range(len(A)):
        if j != i:
            if S is None:
                S =  tenpy.dot(tenpy.transpose(A[j]), A[j])
            else:
                S *= tenpy.dot(tenpy.transpose(A[j]), A[j])
    S += Regu * tenpy.eye(S.shape[0])
    return S

def compute_lin_sys(tenpy, X, Y, Regu):
    return tenpy.dot(tenpy.transpose(X), X) * tenpy.dot(tenpy.transpose(Y), Y) + Regu*tenpy.eye(Y.shape[1])

def randomized_svd(tenpy,A,r,iter=1):
    n = A.shape[1]
    X = tenpy.random((n,r))
    q,_ = tenpy.qr(X)
    ATA = tenpy.dot(tenpy.transpose(A),A)
    for i in range(iter):
        X = tenpy.einsum("jl,lr->jr",ATA,q)
        q,_ = tenpy.qr(X)
        B = tenpy.dot(A,q)
    U,s,VT = tenpy.svd(B)
    VT = tenpy.einsum("ik,jk->ij",VT,q)
    return [U,s,VT]


def compute_number_of_variables(G):
    a = 0
    for matrix in G:
        a += matrix.shape[0]*matrix.shape[1]
    return a


def flatten_Tensor(tenpy,G):
    l = compute_number_of_variables(G)
    g = tenpy.zeros(l)
    start = 0
    for i in range(len(G)):
        end = start + G[i].shape[0]*G[i].shape[1]
        g[start:end] = tenpy.reshape(G[i],-1,order='F')
        start = end
    return g


def reshape_into_matrices(tenpy,x,template):
    ret = []
    start = 0
    for i in range(len(template)):
        end = start + template[i].shape[0]*template[i].shape[1]
        A = tenpy.reshape(x[start:end],template[i].shape,order='F')
        ret.append(A)
        start = end
    return ret


def solve_sys_svd(tenpy, G, RHS):
    t0 = time.time()
    [U,S,VT] = tenpy.svd(G)
    S = 1./S
    X = tenpy.dot(RHS, U)
    0.*X.i("ij") << S.i("j") * X.i("ij")
    t1 = time.time()
    tenpy.printf("Solving linear system took ",t1-t0,"seconds")
    return tenpy.dot(X, VT)

def solve_sys(tenpy, G, RHS):
    L = tenpy.cholesky(G)
    X = tenpy.solve_tri(L, RHS, True, False, True)
    X = tenpy.solve_tri(L, X, True, False, False)
    return X

def get_residual3(tenpy,T,A,B,C):
    t0 = time.time()
    nrm = tenpy.vecnorm(T-tenpy.einsum("ia,ja,ka->ijk",A,B,C))
    t1 = time.time()
    tenpy.printf("Residual computation took",t1-t0,"seconds")
    return nrm

def get_residual_sp3(tenpy,O,T,A,B,C):
    t0 = time.time()
    K = tenpy.TTTP(O,[A,B,C])
    nrm1 = tenpy.vecnorm(K)**2
    tenpy.printf("Non-zero terms have norm ",nrm1**.5)
    nrm2 = tenpy.sum(tenpy.dot(tenpy.transpose(A),A)*tenpy.dot(tenpy.transpose(B),B)*tenpy.dot(tenpy.transpose(C),C))
    #tenpy.einsum("ia,ib,ja,jb,ka,kb->ab",A,A,B,B,C,C))
    diff = T - K
    nrm3 = tenpy.vecnorm(diff)**2
    tenpy.printf("Non-zero residual is ", nrm3**.5)
    nrm = (nrm3+nrm2-nrm1)**.5
    t1 = time.time()
    tenpy.printf("Sparse residual computation took",t1-t0,"seconds")
    return nrm

def get_residual(tenpy,T,A):
    t0 = time.time()
    V = tenpy.ones(T.shape)
    nrm = tenpy.vecnorm(T-tenpy.TTTP(V,A))
    t1 = time.time()
    #tenpy.printf("Residual computation took",t1-t0,"seconds")
    return nrm

def get_residual_sp(tenpy,O,T,A):
    t0 = time.time()
    K = tenpy.TTTP(O,A)
    nrm1 = tenpy.vecnorm(K)**2
    B = tenpy.dot(tenpy.transpose(A[0]),A[0])
    for i in range(1,len(A)):
      B *= tenpy.dot(tenpy.transpose(A[i]),A[i])
    nrm2 = tenpy.sum(B)
    diff = T - K
    nrm3 = tenpy.vecnorm(diff)**2
    nrm = (nrm3+nrm2-nrm1)**.5
    t1 = time.time()
    tenpy.printf("Sparse residual computation took",t1-t0,"seconds")
    return nrm

def equilibrate(tenpy,A):
    nrms = []
    delta = tenpy.ones(A[0].shape[1])
    for i in range(len(A)): 
        nrms.append((tenpy.einsum('ir,ir->r',A[i],A[i]))**0.5)
        delta *= nrms[i]
    delta = delta**(1/len(A))
    for i in range(len(A)):
        A[i] = A[i]*delta/nrms[i]
    return A

def normalise(tenpy,A):
    for i in range(len(A)): 
        A[i] = A[i]/tenpy.einsum('ir,ir->r',A[i],A[i])**0.5
    return A
