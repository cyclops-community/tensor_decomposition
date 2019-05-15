import numpy as np
import sys
import time
from .common_kernels import solve_sys, compute_lin_sys

def naive_ALS_step(tenpy,T,A,B,C,Regu):
    RHS = tenpy.einsum("ijk,ja,ka->ia",T,B,C)
    A = solve_sys(tenpy,compute_lin_sys(tenpy,B,C,Regu), RHS)
    RHS = tenpy.einsum("ijk,ia,ka->ja",T,A,C)
    B = solve_sys(tenpy,compute_lin_sys(tenpy,A,C,Regu), RHS)
    RHS = tenpy.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(tenpy,compute_lin_sys(tenpy,A,B,Regu), RHS)
    return [A,B,C]

def dt_ALS_step(tenpy,T,A,B,C,Regu):
    TC = tenpy.einsum("ijk,ka->ija",T,C)
    RHS = tenpy.einsum("ija,ja->ia",TC,B)
    A = solve_sys(tenpy,compute_lin_sys(tenpy,B,C,Regu), RHS)
    RHS = tenpy.einsum("ija,ia->ja",TC,A)
    B = solve_sys(tenpy,compute_lin_sys(tenpy,A,C,Regu), RHS)
    RHS = tenpy.einsum("ijk,ia,ja->ka",T,A,B)
    C = solve_sys(tenpy,compute_lin_sys(tenpy,A,B,Regu), RHS)
    return [A,B,C]


