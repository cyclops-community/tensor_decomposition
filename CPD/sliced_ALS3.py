import numpy as np
import sys
import time
from .common_kernels import compute_lin_sys

def sliced_ALS_step(Ta,Tb,Tc,A,B,C,Regu):
    L = tenpy.cholesky(compute_lin_sys(B,C,Regu))
    i_st = 0
    i_end = 0
    for i in range(len(Ta)):
      i_st = i_end
      i_end = i_st + Ta[i].shape[0]
      RHS = tenpy.einsum("ijk,ja,ka->ia",Ta[i],B,C)
      X = tenpy.solve_tri(L, RHS, True, False, True)
      A[i_st:i_end,:] = tenpy.solve_tri(L, X, True, False, False)

    L = tenpy.cholesky(compute_lin_sys(A,C,Regu))
    j_st = 0
    j_end = 0
    for i in range(len(Tb)):
      j_st = j_end
      j_end = j_st + Tb[i].shape[1]
      RHS = tenpy.einsum("ijk,ia,ka->ja",Tb[i],A,C)
      X = tenpy.solve_tri(L, RHS, True, False, True)
      B[j_st:j_end,:] = tenpy.solve_tri(L, X, True, False, False)

    L = tenpy.cholesky(compute_lin_sys(A,B,Regu))
    k_st = 0
    k_end = 0
    for i in range(len(Tc)):
      k_st = k_end
      k_end = k_st + Tc[i].shape[2]
      RHS = tenpy.einsum("ijk,ia,ja->ka",Tc[i],A,B)
      X = tenpy.solve_tri(L, RHS, True, False, True)
      C[k_st:k_end,:] = tenpy.solve_tri(L, X, True, False, False)

    return [A,B,C]

