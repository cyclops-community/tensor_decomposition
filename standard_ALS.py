import numpy as np
import queue
import ctf
from ctf import random
import common_ALS3_kernels as cak


def dt_ALS_step(T,A,Regu):
    q = queue.Queue()
    for i in range(len(A)):
        q.put(i)
    s = [(list(range(len(A))),T)]
    while not q.empty():
        i = q.get()
        while i not in s[-1][0]:
            s.pop()
            assert(len(s) >= 1)
        while len(s[-1][0]) != 1:
            M = s[-1][1]
            idx = s[-1][0].index(i)
            ii = len(s[-1][0])-1
            if idx == len(s[-1][0])-1:
                ii = len(s[-1][0])-2
            
            ci = ""
            nd = M.ndim
            if len(s) != 1:
                ci ="R"
                nd = M.ndim-1
            einstr = "".join([chr(ord('a')+j) for j in range(nd)])+ci+"," \
                        + (chr(ord('a')+ii))+"R"+"->"+"".join([chr(ord('a')+j) for j in range(nd) if j != ii])+"R"
            N = ctf.einsum(einstr,M,A[ii])
            ss = s[-1][0].copy()
            ss.remove(ii)
            s.append((ss,N))
        A[i] = cak.solve_sys(cak.compute_lin_sysN(A,i,Regu), s[-1][1])
    return A

