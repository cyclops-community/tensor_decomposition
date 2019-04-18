import numpy as np
import ctf
from ctf import random
import sys
import time

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
