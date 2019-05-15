import numpy as np
import queue
from .common_kernels import solve_sys, compute_lin_sysN

class DTALS_Optimizer(object):

    def __init__(self,tenpy):
        self.tenpy = tenpy
        pass

    def einstr_builder(self,M,s,ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci ="R"
            nd = M.ndim-1
        einstr = "".join([chr(ord('a')+j) for j in range(nd)])+ci+"," \
            + (chr(ord('a')+ii))+"R"+"->"+"".join([chr(ord('a')+j) for j in range(nd) if j != ii])+"R"
        return einstr

    def step(self,tenpy,T,A,Regu):
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
                
                einstr = self.einstr_builder(M,s,ii)

                N = self.tenpy.einsum(einstr,M,A[ii])
                ss = s[-1][0].copy()
                ss.remove(ii)
                s.append((ss,N))
            A[i] = solve_sys(self.tenpy,compute_lin_sysN(tenpy,A,i,Regu), s[-1][1])
        return A

class PPALS_Optimizer(object):

    def __init__(self,tenpy):
        self.tenpy = tenpy
        pass

    def step(self,T,A,Regu):
        """
        TODO (Linjian): implement pairwise perturbation
        """
        pass
