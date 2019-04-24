import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak

def naive_ALS_step(T,A,B,C,Regu):
    RHS = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    A = cak.solve_sys(cak.compute_lin_sys(B,C,Regu), RHS)
    RHS = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    B = cak.solve_sys(cak.compute_lin_sys(A,C,Regu), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = cak.solve_sys(cak.compute_lin_sys(A,B,Regu), RHS)
    return [A,B,C]

def dt_ALS_step(T,A,B,C,Regu):
    TC = ctf.einsum("ijk,ka->ija",T,C)
    RHS = ctf.einsum("ija,ja->ia",TC,B)
    A = cak.solve_sys(cak.compute_lin_sys(B,C,Regu), RHS)
    RHS = ctf.einsum("ija,ia->ja",TC,A)
    B = cak.solve_sys(cak.compute_lin_sys(A,C,Regu), RHS)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = cak.solve_sys(cak.compute_lin_sys(A,B,Regu), RHS)
    return [A,B,C]


