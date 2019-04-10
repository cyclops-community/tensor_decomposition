import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak

def sliced_ALS_step(Ta,Tb,Tc,A,B,C):
    L = ctf.cholesky(cak.compute_lin_sys(B,C))
    i_st = 0
    i_end = 0
    for i in range(len(Ta)):
      i_st = i_end
      i_end = i_st + Ta[i].shape[0]
      RHS = ctf.einsum("ijk,ja,ka->ia",Ta[i],B,C)
      X = ctf.solve_tri(L, RHS, True, False, True)
      A[i_st:i_end,:] = ctf.solve_tri(L, X, True, False, False)

    L = ctf.cholesky(cak.compute_lin_sys(A,C))
    j_st = 0
    j_end = 0
    for i in range(len(Tb)):
      j_st = j_end
      j_end = j_st + Tb[i].shape[1]
      RHS = ctf.einsum("ijk,ia,ka->ja",Tb[i],A,C)
      X = ctf.solve_tri(L, RHS, True, False, True)
      B[j_st:j_end,:] = ctf.solve_tri(L, X, True, False, False)

    L = ctf.cholesky(cak.compute_lin_sys(A,B))
    k_st = 0
    k_end = 0
    for i in range(len(Tc)):
      k_st = k_end
      k_end = k_st + Tc[i].shape[2]
      RHS = ctf.einsum("ijk,ia,ja->ka",Tc[i],A,B)
      X = ctf.solve_tri(L, RHS, True, False, True)
      C[k_st:k_end,:] = ctf.solve_tri(L, X, True, False, False)

    return [A,B,C]


#def dt_ALS_step(Ta,Tb,Tc,A,B,C,b):
#    k = A.shape[1]
#    nb = int((k+b-1)/b)
#    RHS = ctf.tensor((T[0],k))
#    G = cak.compute_lin_sys(B,C)
#    L = ctf.cholesky(G)
#    for i in range(nb):
#      st = i*b
#      end = min(i*b,k)
#      Ci = C[:,st:end]
#      Bi = B[:,st:end]
#      TC = ctf.einsum("ijk,ka->ija",T,Ci)
#      RHS[:,st:end] = ctf.einsum("ija,ja->ia",TC,Bi)
#      X = ctf.solve_tri(L, RHS, True, False, True)
#      X = ctf.solve_tri(L, X, True, False, False)
#      A = cak.solve_sys(, RHS)
#    for i in range(nb):
#      st = i*b
#      end = min(i*b,k)
#      Ai = A[:,st:end]
#      Bi = B[:,st:end]
#    RHS = ctf.einsum("ija,ia->ja",TC,A)
#    B = cak.solve_sys(cak.compute_lin_sys(A,C), RHS)
#    RHS = ctf.einsum("ijk,ia,ja->ka",T,A,B)
#    C = cak.solve_sys(cak.compute_lin_sys(A,B), RHS)
#    return [A,B,C]
#
#
