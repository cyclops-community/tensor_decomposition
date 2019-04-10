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


