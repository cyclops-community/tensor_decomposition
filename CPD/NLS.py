from CPD.common_kernels import compute_number_of_variables, flatten_Tensor, reshape_into_matrices, solve_sys, get_residual
from scipy.sparse.linalg import LinearOperator
from CPD.standard_ALS import CP_DTALS_Optimizer
 

import scipy.sparse.linalg as spsalg
import numpy as np
import time

try:
    import Queue as queue
except ImportError:
    import queue

def fast_hessian_contract(tenpy,X,A,gamma,regu=1):
    N = len(A)
    ret = []
    for n in range(N):
        ret.append(tenpy.zeros(A[n].shape))
        for p in range(N):
            M = gamma[n][p]
            if n==p:
                ret[n] += tenpy.einsum("iz,zr->ir",X[p],M)
            else:
                time0 = time.time()
                B = tenpy.einsum("jr,jz->rz",A[p],X[p])
                time1 = time.time()
                ret[n] += tenpy.einsum("iz,zr,rz->ir",A[n],M,B)
                time2 = time.time()
                #tenpy.printf("compute B took ",time1-time0," seconds.")
                #tenpy.printf("compute ret took ",time2-time1,"seconds.")
                #Y = tenpy.einsum("iz,zr,jr,jz->ir",A[n],M,A[p],X[p])

        ret[n] += regu*X[n]

    return ret

def fast_block_diag_precondition(tenpy,X,P):
    N = len(X)
    ret = []
    for i in range(N):
        Y = tenpy.solve_tri(P[i], X[i], True, False, True)
        Y = tenpy.solve_tri(P[i], Y, True, False, False)
        ret.append(Y)
    return ret

class CP_fastNLS_Optimizer():
    """Fast Nonlinear Least Square Method for CP is a novel method of
    computing the CP decomposition of a tensor by utilizing tensor contractions
    and preconditioned conjugate gradient to speed up the process of solving
    damped Gauss-Newton problem of CP decomposition.
    """

    def __init__(self,tenpy,T,A,cg_tol=1e-4,num=1,args=None):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.cg_tol = cg_tol
        self.num=num
        self.G = None
        self.gamma = None
        #self.side_length = get_side_length(A)
        #self.last_step = tenpy.zeros((compute_sum_side_length(A),A[0].shape[1]))
        self.atol = 0


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

    def compute_G(self):
        G = []
        for i in range(len(self.A)):
            G.append(self.tenpy.einsum("ij,ik->jk",self.A[i],self.A[i]))
        self.G = G

    def compute_coefficient_matrix(self,n1,n2):
        ret = self.tenpy.ones(self.G[0].shape)
        for i in range(len(self.G)):
            if i!=n1 and i!=n2:
                ret = self.tenpy.einsum("ij,ij->ij",ret,self.G[i])
        return ret

    def compute_gamma(self):
        N = len(self.A)
        result = []
        for i in range(N):
            result.append([])
            for j in range(N):
                if j>=i:
                    M = self.compute_coefficient_matrix(i,j)
                    result[i].append(M)
                else:
                    M = result[j][i]
                    result[i].append(M)
        self.gamma = result

    def compute_block_diag_preconditioner(self,Regu):
        P = []
        for i in range(len(self.A)):
            n = self.gamma[i][i].shape[0]
            P.append(self.tenpy.cholesky(self.gamma[i][i]+Regu*self.tenpy.eye(n)))
        return P


    def gradient(self):
        grad = []
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

                einstr = self._einstr_builder(M,s,ii)

                N = self.tenpy.einsum(einstr,M,self.A[ii])

                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss,N))
            M = s[-1][1]
            g = -1*M + self.A[i].dot(self.gamma[i][i])
            grad.append(g)
        return grad

    def power_method(self,l,iter=1):
        for i in range(iter):
            l = fast_hessian_contract(self.tenpy,l,self.A,self.gamma,0)
            a = self.tenpy.list_vecnorm(l)
            l = self.tenpy.scalar_mul(1/a,l)
        return l

    def rayleigh_quotient(self,l):
        a = self.tenpy.mult_lists(l,fast_hessian_contract(self.tenpy,l,self.A,self.gamma,0))
        b = self.tenpy.list_vecnormsq(l)
        return a/b


    def create_fast_hessian_contract_LinOp(self,Regu):
        num_var = compute_number_of_variables(self.A)
        A = self.A
        gamma = self.gamma
        tenpy = self.tenpy
        template = self.A

        def mv(delta):
            delta = reshape_into_matrices(tenpy,delta,template)
            result = fast_hessian_contract(tenpy,delta,A,gamma,Regu)
            vec = flatten_Tensor(tenpy,result)
            return vec

        V = LinearOperator(shape = (num_var,num_var), matvec=mv)
        return V

    def matvec(self,Regu,delta):
        #t = ctf.time("total fast hessian multiplication")
        t0 = time.time()
        #t.start()
        A = self.A
        gamma = self.gamma
        tenpy = self.tenpy
        template = self.A
        result = fast_hessian_contract(tenpy,delta,A,gamma,Regu)
        t1 = time.time()
        #t.end()
        #self.tenpy.printf("fast hessian contract took ",t1-t0,"seconds.")
        return result

    def fast_conjugate_gradient(self,g,Regu):

        x = [self.tenpy.zeros(A.shape) for A in g]

        tol = np.max([self.atol,self.cg_tol*self.tenpy.list_vecnorm(g)])


        r = self.tenpy.list_add(self.tenpy.scalar_mul(-1,g), self.tenpy.scalar_mul(-1,self.matvec(Regu,x)))

        if self.tenpy.list_vecnorm(r)<tol:
            return x
        p = r
        counter = 0

        while True:
            mv = self.matvec(Regu,p)

            alpha = self.tenpy.list_vecnormsq(r)/self.tenpy.mult_lists(p,mv)

            x = self.tenpy.list_add(x,self.tenpy.scalar_mul(alpha,p))

            r_new = self.tenpy.list_add(r, self.tenpy.scalar_mul(-1,self.tenpy.scalar_mul(alpha,mv)))

            if self.tenpy.list_vecnorm(r_new)<tol:
                break
            beta = self.tenpy.list_vecnormsq(r_new)/self.tenpy.list_vecnormsq(r)

            p = self.tenpy.list_add(r_new, self.tenpy.scalar_mul(beta,p))
            r = r_new
            counter += 1


        return x,counter

    def fast_precond_conjugate_gradient(self,g,P,Regu):
        x = [self.tenpy.zeros(A.shape) for A in g]

        tol = np.max([self.atol,self.cg_tol*self.tenpy.list_vecnorm(g)])

        r = self.tenpy.list_add(self.tenpy.scalar_mul(-1,g), self.tenpy.scalar_mul(-1,self.matvec(Regu,x)))

        if self.tenpy.list_vecnorm(r)<tol:
            return x

        z = fast_block_diag_precondition(self.tenpy,r,P)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu,p)

            mul = self.tenpy.mult_lists(r,z)

            alpha = mul/self.tenpy.mult_lists(p,mv)

            x =self.tenpy.list_add(x,self.tenpy.scalar_mul(alpha,p))

            r_new = self.tenpy.list_add(r, self.tenpy.scalar_mul(-1,self.tenpy.scalar_mul(alpha,mv)))

            if self.tenpy.list_vecnorm(r_new)<tol:
                break

            z_new = fast_block_diag_precondition(self.tenpy,r_new,P)

            #if formula == "PR":
            #    beta = np.inner(z_new,r_new-r)/np.inner(z,r)
            #else:
            beta = self.tenpy.mult_lists(r_new,z_new)/mul

            p = self.tenpy.list_add(z_new, self.tenpy.scalar_mul(beta,p))

            r = r_new
            z = z_new
            counter += 1

        return x,counter

    def create_block_precondition_LinOp(self,P):
        num_var = compute_number_of_variables(self.A)
        tenpy = self.tenpy
        template = self.A

        def mv(delta):

            delta = reshape_into_matrices(tenpy,delta,template)
            result = fast_block_diag_precondition(tenpy,delta,P)
            vec = flatten_Tensor(tenpy,result)
            return vec

        V = LinearOperator(shape = (num_var,num_var), matvec=mv)
        return V



    def update_A(self,delta):
        for i in range(len(delta)):
            self.A[i] += delta[i]



    def step2(self,Regu):
        self.compute_G()
        self.compute_gamma()
        g= self.gradient()

        P = self.compute_block_diag_preconditioner(Regu)

        #[delta,counter] = self.fast_conjugate_gradient(g,Regu)

        [delta,counter] = self.fast_precond_conjugate_gradient(g,P,Regu)

        self.atol = self.num*self.tenpy.list_vecnorm(delta)
        self.tenpy.printf('cg iterations:',counter)
        self.update_A(delta)

        return self.A



    def step(self,Regu):
        """
        """
        global cg_iters
        cg_iters =0

        def cg_call(v):
            global cg_iters
            cg_iters= cg_iters+1

        self.compute_G()
        self.compute_gamma()
        #l = self.power_method([self.tenpy.random(M.shape) for M in self.A])
        #L2 = self.rayleigh_quotient(l)
        #print("L2 norm of hessian is ",L2)
        g = flatten_Tensor(self.tenpy,self.gradient())
        mult_LinOp = self.create_fast_hessian_contract_LinOp(Regu)
        P = self.compute_block_diag_preconditioner(Regu)
        precondition_LinOp = self.create_block_precondition_LinOp(P)
        [delta,_] = spsalg.cg(mult_LinOp,-1*g,tol=self.cg_tol,M=precondition_LinOp,callback=cg_call,atol=self.atol)
        #[delta,_] = spsalg.cg(mult_LinOp,-1*g,tol=self.cg_tol,callback=cg_call,atol=self.atol)
        self.atol = self.num*self.tenpy.norm(delta)
        delta = reshape_into_matrices(self.tenpy,delta,self.A)
        self.update_A(delta)
        self.tenpy.printf('cg iterations:',cg_iters)
        
        return self.A

class CP_ALSNLS_Optimizer(CP_fastNLS_Optimizer,CP_DTALS_Optimizer):

    def __init__(self,tenpy,T,A,cg_tol=1e-04,num=0,switch_tol= 0.1, args=None):
        CP_fastNLS_Optimizer.__init__(self,tenpy,T,A,cg_tol,num,args)
        CP_DTALS_Optimizer.__init__(self,tenpy,T,A)
        self.tenpy = tenpy
        self.switch_tol = switch_tol
        self.switch = False
        self.A = A
        self.count = 0
        self.count_tol = 50

    def _step_dt(self,Regu):
        return CP_DTALS_Optimizer.step(self,Regu)

        
    def _step_nls(self,Regu):
        
        return CP_fastNLS_Optimizer.step2(self,Regu)



    def copy_A(self):
        A_copy = []
        for i in range(len(self.A)):
            A_copy+=[self.A[i].copy()]
        
        return A_copy
        
        
    def step(self,Regu):
        if self.switch:
            self.tenpy.printf("performing nls")
            self.A = self._step_nls(Regu)

        else:
            if self.count != 0:
                A_prev= self.copy_A()
                
                self.A = self._step_dt(Regu)
                self.count+=1
                if self.count > self.count_tol:
                    self.switch = True
                    self.tenpy.printf("count reached, will switch to nls")

                if self.tenpy.list_vecnorm(self.tenpy.list_add(self.A, self.tenpy.scalar_mul(-1,A_prev)))<self.switch_tol:
                    self.switch = True
                    self.tenpy.printf("norm reached, will switch to nls")
            else:
                self.A = self._step_dt(Regu)
                self.count += 1

        return self.A
        
        
class CP_safeNLS_Optimizer(CP_fastNLS_Optimizer,CP_DTALS_Optimizer):

    def __init__(self,tenpy,T,A,cg_tol=1e-04,num=0,als_iter= 10, nls_iter = 2, args=None):
        CP_fastNLS_Optimizer.__init__(self,tenpy,T,A,cg_tol,num,args)
        CP_DTALS_Optimizer.__init__(self,tenpy,T,A)
        self.tenpy = tenpy
        self.switch = True
        self.A = A
        self.T=T
        self.count = 0
        self.als_iter = als_iter
        self.prev_res = get_residual(tenpy,T,A)
        self.nls_steps= 0
        self.nls_iter = nls_iter
        
        
        

    def _step_dt(self,Regu):
        return CP_DTALS_Optimizer.step(self,Regu)



    def update_A(self,delta):
        for i in range(len(delta)):
            self.A[i] += delta[i]
            
            
    def _step_nls(self,Regu):
        
        return CP_fastNLS_Optimizer.step2(self,Regu)
        
     
     
    def copy_A(self):
        A_copy = []
        for i in range(len(self.A)):
            A_copy+=[self.A[i].copy()]
        
        return A_copy
        

    def step(self,Regu):
        if self.switch:
            
            self.count = 0
            
            A_prev =self.copy_A()
            
            self.tenpy.printf("performing nls")
            
            for i in range(self.nls_iter):
                self.A = self._step_nls(Regu)
            
            curr_res = get_residual(self.tenpy,self.T,self.A)
            
            if curr_res<= self.prev_res:
                self.prev_res = curr_res
                self.nls_steps+=1
                
            else:
                self.tenpy.printf("Residual increased for NLS, will switch to ALS")
                self.switch = False
                self.A = A_prev[:] 
                
            

        else:
            delta = self._step_dt(Regu)
            self.count+=1
            self.prev_res = get_residual(self.tenpy,self.T,self.A)
                
            if self.count == self.als_iter:
                self.switch = True
                self.tenpy.printf("count reached, will switch to NLS")




        self.tenpy.printf("number of nls steps performed", self.nls_steps)
        return self.A
