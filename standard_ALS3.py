import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak

def naive_ALS_step(T,A,B,C,Regu):
    [A,_] = cak.norm_cols(A)
    [B,_] = cak.norm_cols(B)
    [C,_] = cak.norm_cols(C)
    RHS = ctf.einsum("ijk,ja,ka->ia",T,B,C)
    A = cak.solve_sys(cak.compute_lin_sys(B,C,Regu), RHS)
    [A,_] = cak.norm_cols(A)
    RHS = ctf.einsum("ijk,ia,ka->ja",T,A,C)
    B = cak.solve_sys(cak.compute_lin_sys(A,C,Regu), RHS)
    [B,_] = cak.norm_cols(B)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = cak.solve_sys(cak.compute_lin_sys(A,B,Regu), RHS)
    return [A,B,C]

def dt_ALS_step(T,A,B,C,Regu):
    [A,_] = cak.norm_cols(A)
    [B,_] = cak.norm_cols(B)
    [C,_] = cak.norm_cols(C)
    TC = ctf.einsum("ijk,ka->ija",T,C)
    RHS = ctf.einsum("ija,ja->ia",TC,B)
    A = cak.solve_sys(cak.compute_lin_sys(B,C,Regu), RHS)
    [A,_] = cak.norm_cols(A)
    RHS = ctf.einsum("ija,ia->ja",TC,A)
    B = cak.solve_sys(cak.compute_lin_sys(A,C,Regu), RHS)
    [B,_] = cak.norm_cols(B)
    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
    C = cak.solve_sys(cak.compute_lin_sys(A,B,Regu), RHS)
    return [A,B,C]


