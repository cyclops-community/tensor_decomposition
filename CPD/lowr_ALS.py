from .common_kernels import solve_sys, compute_lin_sysN
from als.ALS_optimizer import DTALS_base, DTLRALS_base, PPALS_base
from .standard_ALS import CP_DTALS_Optimizer
from .lowr_ALS3 import solve_sys_lowr

class CP_DTLRALS_Optimizer(DTLRALS_base, CP_DTALS_Optimizer):
    """Dimension tree with low rank update CP decomposition optimizer
    """

    def __init__(self,tenpy,T,A,args):
        DTLRALS_base.__init__(self,tenpy,T,A,args)
        CP_DTALS_Optimizer.__init__(self,tenpy,T,A)

    def _step_dt_subroutine(self,Regu):
        return CP_DTALS_Optimizer.step(self,Regu)

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

    def _einstr_builder_lr(self,M,s,ii):
        """Build string used in einsum when applying low rank update.

        Args:
            M (tenpy tensor): the tensor on the current node of the dimension tree
            s (list): a list of nodes of the dimension tree
            ii (int): the current index we need to contract.

        Returns:
            string: the string to be used in einsum.
        """
        assert(len(s)>0)
        if len(s) > 2:
            ci = "LR"
            nd = M.ndim-2
        elif len(s) == 2:
            ci = "L"
            nd = M.ndim-1
        else:
            ci = ""
            nd = M.ndim

        str1 = "".join([chr(ord('a')+j) for j in s[-1][0]])+ci
        str2 = (chr(ord('a')+ii))
        str3 = "".join([chr(ord('a')+j) for j in s[-1][0] if j != ii])

        if len(s) >= 2:
            str3 += "LR"
            str2 += "R"
        else:
            str3 += "L"
            str2 += "L"

        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve_DTLR(self,i,Regu):
        """Get the best low rank update for factor matrix i.

        Args:
            i (int): the index of factor matrix to get low rank update on.
            Regu (tenpy matrix): regularization factor matrix

        Returns:
            U,VT (tenpy matrix): low rank update matrices for factor matrix i.
        """
        G = compute_lin_sysN(self.tenpy,self.A,i,Regu)
        ERHS = self.RHS[i] - self.tenpy.dot(self.A[i],G)
        return solve_sys_lowr(self.tenpy,G,ERHS,self.r)

    def _solve_by_full_rank(self,i,Regu):
        """Get the low rank update for factor matrix by computing the full rank update
        and then truncate.

        Args:
            i (int): the index of factor matrix to get low rank update on.
            Regu (tenpy matrix): regularization factor matrix

        Returns:
            U,VT (tenpy matrix): low rank update matrices for factor matrix i.
        """
        A_new = solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), self.RHS[i])
        dA = A_new - self.A[i]
        [U,S,VT] = self.tenpy.svd(dA,self.r)
        return [U,S,VT]

    def _get_index_by_tol(self,s,tol):
        """Get the end index of the singular values that we want to keep by pre-specified tolerance.

        Args:
            s (tenpy array): a vector of singular values
            tol (float): tolerance

        Returns:
            int: end index of the singular values that we want to keep
        """
        n = s.shape[0]
        i = n-1
        a = 0
        while a<tol and i>=0:
            a += s[i]
            i -= 1
        return i+2

    def _solve_LR_by_tol(self,i,Regu,tol):
        U,s,VT = self._solve_by_full_rank(i,Regu)
        end = self._get_index_by_tol(s,tol)
        U = U[:,:end]
        s = s[:end]
        VT = VT[:end,:]
        VT = self.tenpy.einsum("i,ij->ij",s,VT)
        return U,VT
