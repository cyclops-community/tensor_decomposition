import numpy as np
import ctf
from ctf import random
import sys
import time

def init_rand(order,s,R,sp_frac=1.):
    A = []
    for i in range(order):
        A.append(ctf.random.random((s,R)))
    if sp_frac<1.:
        O = ctf.tensor([s]*order,sp=True)
        O.fill_sp_random(1.,1.,sp_frac)
        T = ctf.TTTP(O,A)
    else:
        T = ctf.ones([s]*order)
        T = ctf.TTTP(T,A)
        O = None
    return [T,O]

def init_rand3(s,R,sp_frac=1.):
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

def init_mm(s,R):
    sr = int(np.sqrt(s)+.1)
    assert(sr*sr==s)
    T = ctf.tensor((sr,sr,sr,sr,sr,sr),sp=True)
    F = ctf.tensor((sr,sr,sr,sr),sp=True)
    I = ctf.speye(sr)
    #F.i("ijkl") << I.i("ik")*I.i("jl");
    #ctf.einsum("ijab,klbc,mnca->ijklmn",F,F,F,out=T)
    ctf.einsum("ia,jb,kb,lc,mc,na->ijklmn",I,I,I,I,I,I,out=T)
    T = T.reshape((s,s,s))
    O = T
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    return [A,B,C,T,O]

def init_poisson(s,R):
    sr = int(s**(1./2)+.1)
    assert(sr*sr==s)
    T = ctf.tensor((sr,sr,sr,sr,sr,sr),sp=True)
    A = (-2.)*ctf.eye(sr,sr,sp=True) + ctf.eye(sr,sr,1,sp=True) + ctf.eye(sr,sr,-1,sp=True)
    I = ctf.eye(sr,sr,sp=True) # sparse identity matrix
    T.i("aixbjy") << A.i("ab")*I.i("ij")*I.i("xy") + I.i("ab")*A.i("ij")*I.i("xy") + I.i("ab")*I.i("ij")*A.i("xy")
    # T.i("abijxy") << A.i("ab")*I.i("ij")*I.i("xy") + I.i("ab")*A.i("ij")*I.i("xy") + I.i("ab")*I.i("ij")*A.i("xy")
    N = ctf.tensor((s,s,s),sp=True)
    N.fill_sp_random(-0.000,.000,1./s)
    T = T.reshape((s,s,s)) + N
    [inds, vals] = T.read_local()
    vals[:] = 1.
    O = ctf.tensor(T.shape,sp=True)
    O.write(inds,vals)
    A = ctf.random.random((s,R))
    B = ctf.random.random((s,R))
    C = ctf.random.random((s,R))
    return [A,B,C,T,O]

def init_mom_cons(k):
    order = 4
    mode_weights = [1, 1, -1, -1]
    
    delta = ctf.tensor(k*np.ones(order))
    [inds,vals] = delta.read_local()
    new_inds = []
    for i in range(len(inds)):
        kval = 0
        ind = inds[i]
        iinds = []
        for j in range(order):
            ind_i = ind % k
            iinds.append(ind_i)
            ind   = ind // k
            kval += mode_weights[-j]*ind_i
        if kval % k == 0:
            new_inds.append(inds[i])
    delta.write(new_inds, np.ones(len(new_inds)))
    return delta


def init_mom_cons_sv(k):
    order = 4
    mode_weights = [1, 1, -1, -1]
    
    delta = ctf.tensor(k*np.ones(order))
    [inds,vals] = delta.read_local()
    new_inds = []
    for i in range(len(inds)):
        kval = 0
        ind = inds[i]
        iinds = []
        for j in range(order):
            ind_i = ind % k
            iinds.append(ind_i)
            ind   = ind // k
            kval += mode_weights[-j]*ind_i
        if kval % k == 0:
            new_inds.append(inds[i])
    delta.write(new_inds, np.ones(len(new_inds)))
    [U,S,VT]=delta.i("ijkl").svd("ija","akl",threshold=1.e-3)
    return U
