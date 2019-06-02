import numpy as np
from .common_kernels import solve_sys, compute_lin_sysN
from als.ALS_optimizer import DTALS_base, PPALS_base, partialPP_ALS_base

class CP_DTALS_Optimizer(DTALS_base):

    def _einstr_builder(self,M,s,ii):
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

    def _solve(self,i,Regu,s):
        return solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), s[-1][1])


class CP_PPALS_Optimizer(PPALS_base, CP_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """

    def __init__(self,tenpy,T,A,args):
        PPALS_base.__init__(self,tenpy,T,A,args)
        CP_DTALS_Optimizer.__init__(self,tenpy,T,A)

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction.

        This function contract the tensor represented by the parent_nodeindex and
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum

        Example:
            When the input tensor has 4 dimensions:
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcR,cR->abR"

        """
        ci = ""
        if len(parent_nodeindex) != self.order:
            ci = "R"

        str1 = "".join([chr(ord('a')+j) for j in parent_nodeindex]) + ci
        str2 = (chr(ord('a')+contract_index)) + "R"
        str3 = "".join([chr(ord('a')+j) for j in nodeindex]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _step_dt(self,Regu):
        return CP_DTALS_Optimizer.step(self,Regu)

    def _solve_PP(self,i,Regu,N):
        return solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), N)

class CP_partialPPALS_Optimizer(partialPP_ALS_base, CP_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """

    def __init__(self,tenpy,T,A,args):
        partialPP_ALS_base.__init__(self,tenpy,T,A,args)
        CP_DTALS_Optimizer.__init__(self,tenpy,T,A)

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction.

        This function contract the tensor represented by the parent_nodeindex and
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum

        Example:
            When the input tensor has 4 dimensions:
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcR,cR->abR"

        """
        ci = ""
        if len(parent_nodeindex) != self.order:
            ci = "R"

        str1 = "".join([chr(ord('a')+j) for j in parent_nodeindex]) + ci
        str2 = (chr(ord('a')+contract_index)) + "R"
        str3 = "".join([chr(ord('a')+j) for j in nodeindex]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _step_dt(self,Regu):
        return CP_DTALS_Optimizer.step(self,Regu)

    def _solve_PP(self,i,Regu,N):
        return solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), N)
