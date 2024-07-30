import numpy as np
import queue
from .common_kernels import n_mode_eigendec
from als.ALS_optimizer import DTALS_base, PPALS_base


class Tucker_DTALS_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A):
        DTALS_base.__init__(self, tenpy, T, A)
        self.tucker_rank = []
        for i in range(len(A)):
            self.tucker_rank.append(A[i].shape[1])

    def _einstr_builder(self, M, s, ii):
        nd = M.ndim

        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = (chr(ord('a') + ii)) + "R"
        str3 = "".join([chr(ord('a') + j) for j in range(ii)]) + "R" + "".join(
            [chr(ord('a') + j) for j in range(ii + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve(self, i, Regu, s):
        # NOTE: Regu is not used here
        return n_mode_eigendec(self.tenpy,
                               s[-1][1],
                               i,
                               rank=self.tucker_rank[i],
                               do_flipsign=True)


class Tucker_PPALS_Optimizer(PPALS_base, Tucker_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """
    def __init__(self, tenpy, T, A, args):
        PPALS_base.__init__(self, tenpy, T, A, args)
        Tucker_DTALS_Optimizer.__init__(self, tenpy, T, A)

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
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcd,cR->abRd"

        """
        nd = self.order
        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = (chr(ord('a') + contract_index)) + "R"
        str3 = "".join(
            [chr(ord('a') + j)
             for j in range(contract_index)]) + "R" + "".join(
                 [chr(ord('a') + j) for j in range(contract_index + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _step_dt(self, Regu):
        return Tucker_DTALS_Optimizer.step(self, Regu)

    def _solve_PP(self, i, Regu, N):
        return n_mode_eigendec(self.tenpy, N, i, rank=self.R, do_flipsign=True)
