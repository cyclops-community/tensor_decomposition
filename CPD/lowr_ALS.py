import queue
from .common_kernels import solve_sys, compute_lin_sysN
from als.ALS_optimizer import DTALS_base, PPALS_base
from .lowr_ALS3 import solve_sys_lowr

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

    def _einstr_builder_lr(self,M,s,ii):
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

        str1 = "".join([chr(ord('a')+j) for j in range(nd)])+ci
        str2 = (chr(ord('a')+ii))+"R"
        str3 = "".join([chr(ord('a')+j) for j in range(nd) if j != ii])

        if len(s) >= 2:
            str3 += "LR"
        else:
            str3 += "L"

        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve(self,i,Regu,RHS):
        return solve_sys_lowr(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), RHS)
