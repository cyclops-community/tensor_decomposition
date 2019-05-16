import numpy as np
import queue
from .common_kernels import solve_sys, compute_lin_sysN

class DTALS_Optimizer(object):

    def __init__(self,tenpy,T,A):
        self.tenpy = tenpy
        self.T = T
        self.A = A

    def einstr_builder(self,M,s,ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci ="R"
            nd = M.ndim-1

        str1 = "".join([chr(ord('a')+j) for j in range(nd)])+ci
        str2 = (chr(ord('a')+ii))+"R"
        str3 = "".join([chr(ord('a')+j) for j in range(nd) if j != ii])+"R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def step(self,Regu):
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))),self.T)]
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

                N = self.tenpy.einsum(einstr,M,self.A[ii])
                ss = s[-1][0].copy()
                ss.remove(ii)
                s.append((ss,N))
            self.A[i] = solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), s[-1][1])
        return self.A



class PPALS_Optimizer(DTALS_Optimizer):

    def __init__(self,tenpy,T,A,tol_restart_dt):

        super(PPALS_Optimizer, self).__init__(tenpy,T,A)
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = tol_restart_dt
        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.order = len(A)
        self.dA = []  #TODO: set all elements to be 0

    def _get_nodename(self, nodeindex):
        """TODO
        """
        return nodename

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """TODO
        """
        return einstr

    def _get_parentnode(self, nodeindex):
        """TODO
        """
        return parent_nodename, parent_nodeindex, contract_index

    def _initialize_treenode(self, nodeindex):

        nodename = self._get_nodename(nodeindex)
        parent_nodename, parent_nodeindex, contract_index = self._get_parentnode(nodeindex)
        einstr = self._get_einstr(nodeindex,parent_nodeindex,contract_index)

        if not self.tree.find(parent_nodename):
            self._initialize_treenode(parent_nodeindex)

        N = self.tenpy.einsum(einstr,self.tree[parent_nodename][1],self.A[contract_index])

        self.tree[nodename] = (nodeindex,N)

    def _initialize_tree(self):

        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.dA = []  #TODO: set all elements to be 0

        for ii in range(0, order):
            for jj in range(ii+1, order):
                self._initialize_treenode([ii,jj])

        for ii in range(0, order):
            self._initialize_treenode([ii])

    def _step_pp_subroutine(self,Regu):
        """TODO
        """
        for i in range(order):
            nodename = self._get_nodename([i])
            N = self.tree[nodename]

            for j in range(i):
                parentname = self._get_nodename([j,i])
                #TODO: implement einstr
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])
            for j in range(i+1, order):
                parentname = self._get_nodename([i,j])
                #TODO: implement einstr
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])

            output = solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), N)
            self.dA[i] = output - self.A[i]
            self.A = output

        #TODO: update self.pp
        return self.A


    def _step_dt_subroutine(self,Regu):

        A = super(PPALS_Optimizer, self).step(Regu)
        #TODO: update self.pp and self.reinitialize_tree
        return A

    def step(self,Regu):
        """
        TODO (Linjian): implement pairwise perturbation
        """
        if self.pp:
            if self.reinitialize_tree:
                self._initialize_tree()
            A = self._step_pp_subroutine(Regu)
        else:
            A = self._step_dt_subroutine(Regu)
        return A
