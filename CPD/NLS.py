from CPD.common_kernels import compute_number_of_variables,  flatten_Tensor, reshape_into_matrices,  get_residual
from scipy.sparse.linalg import LinearOperator

import scipy.sparse.linalg as spsalg
import numpy as np

try:
    import Queue as queue
except ImportError:
    import queue

def fast_hessian_contract_batch(tenpy,XX,AA,GG,GD,regu=1):
    RR  = regu*XX
    RR += tenpy.einsum("niz,nzr->nir",XX,GD)
    if tenpy.name() == 'numpy':
        RR += np.einsum("niz,pjr,npzr,pjz->nir",AA,AA,GG,XX,optimize=True)
    else:
        RR += tenpy.einsum("niz,pjr,npzr,pjz->nir",AA,AA,GG,XX)
    return RR


def fast_hessian_contract(tenpy,X,A,gamma,diag,regu=1):
    N = len(A)
    ret = []
    for n in range(N):
        ret.append(tenpy.zeros(A[n].shape))
        for p in range(N):
            if n==p:
                ret[n] += tenpy.einsum("iz,zr->ir",X[p],gamma[n][p])
            else:
                ret[n] += tenpy.einsum("iz,zr,jr,jz->ir",A[n],gamma[n][p],A[p],X[p])
        
        if diag:
            ret[n] += regu*tenpy.einsum('jj,ij->ij',gamma[n][n],X[n])
            
        else:
            ret[n]+= regu*X[n]
            
    return ret

def fast_block_diag_precondition(tenpy,X,P):
    N = len(X)
    ret = []
    for i in range(N):
        # Y = tenpy.solve_tri(P[i], X[i], True, False, True)
        # Y = tenpy.solve_tri(P[i], Y, True, False, False)
        Y = tenpy.dot(X[i], P[i])
        ret.append(Y)
    return ret

class CP_fastNLS_Optimizer():
    """Fast Nonlinear Least Square Method for CP is a novel method of
    computing the CP decomposition of a tensor by utilizing tensor contractions
    and preconditioned conjugate gradient to speed up the process of solving
    damped Gauss-Newton problem of CP decomposition.
    """

    def __init__(self,tenpy,T,A,args=None):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.cg_tol = args.cg_tol
        self.num=args.num
        self.G = None
        self.gamma = None
        self.atol = 0
        self.total_iters = 0
        self.maxiter = args.maxiter
        self.nls_iter = 0
        self.g_norm = 0
        self.temp = None
        self.GG = None
        self.GD = None
        self.diag = args.diag
        self.Arm = args.arm
        self.c = args.c
        self.tau=args.tau
        self.arm_iter= args.arm_iters
        self.delta = None
        self.sp = args.sp
        


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
        
    def return_gamma(self):
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
        return result
        

    
    def compute_block_diag_preconditioner(self,Regu):
        P = []
        for i in range(len(self.A)):
            n = self.gamma[i][i].shape[0]
            if self.diag:
                U, s, V = self.tenpy.svd(self.gamma[i][i]+Regu*self.tenpy.diag(self.gamma[i][i].diagonal()))
            else:
                U, s, V = self.tenpy.svd(self.gamma[i][i]+Regu*self.tenpy.eye(n))
            Pval_inv = self.tenpy.dot(
                self.tenpy.transpose(V), 
                self.tenpy.dot(
                    self.tenpy.diag(s**-1), self.tenpy.transpose(U)
                    )
                )
            P.append(Pval_inv)
        return P


    def gradient(self):
        grad = []
        if self.sp:
            lst = self.A[:]
            for i in range(len(self.A)):
                out = self.tenpy.tensor(self.A[i].shape,sp=1.)
                lst[i] = out 
                self.tenpy.MTTKRP(self.T,lst,i)
                grad.append(lst[i] - self.A[i].dot(self.gamma[i][i]))
                lst[i] = self.A[i]

        else:
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
                g = M - self.A[i].dot(self.gamma[i][i])
                grad.append(g)
        return grad
    
    def gradient_GG(self,g):
        N = len(self.A)
        s = self.A[0].shape[0]
        R = self.A[0].shape[1]
        
        gg = self.tenpy.zeros((N,s,R))
        
        self.AA = self.tenpy.zeros((N,s,R))
        self.GG = self.tenpy.zeros((N,N,R,R))
        self.GD = self.tenpy.zeros((N,R,R))
        
        for n in range(N):
            self.AA[n,:,:] = self.A[n]
            gg[n,:,:] = g[n]
            for p in range(N):
                if n != p:
                    self.GG[n,p,:,:] = self.gamma[n][p]
                else:
                    self.GD[n,:,:] = self.gamma[n][n]
       
        
        return gg
    


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
        return fast_hessian_contract(self.tenpy,delta,self.A,self.gamma,self.diag,Regu)
    
    def matvec2(self,Regu,delta):
        return fast_hessian_contract_batch(self.tenpy,delta,self.AA,self.GG,self.GD,Regu)

    def fast_conjugate_gradient_batch(self,gg,Regu):
        #start = time.time()

        N = len(self.A)
        XX = self.tenpy.zeros(self.AA.shape)
        
        tol = np.max([self.atol,self.cg_tol*self.tenpy.vecnorm(gg)])

        RR = gg

        if self.tenpy.vecnorm(RR)<tol:
            return XX
        PP = RR
        counter = 0

        while True:
            MV = self.matvec2(Regu,PP)
            alpha = self.tenpy.vecnorm(RR)**2./self.tenpy.dot_product(PP,MV)

            XX += alpha*PP

            RR_new = RR - alpha*MV

            if self.tenpy.vecnorm(RR_new)<tol:
                counter+=1
                #end = time.time()
                break

            beta = self.tenpy.vecnorm(RR_new)**2/(self.tenpy.vecnorm(RR)**2)

            PP = RR_new + beta*PP
            RR = RR_new
            counter += 1

            if counter == self.maxiter:
                #end = time.time()
                break
                
        #self.tenpy.printf('cg took',end-start)
        x = [XX[i,:,:] for i in range(N)]

        return x,counter


    def fast_conjugate_gradient(self,g,Regu):
        #start = time.time()

        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        
        r = g
        
        #self.tenpy.printf('starting res in cg is',self.tenpy.list_vecnorm(r))
        if g_norm <tol:
            return x
        
        p = r
        counter = 0

        while True:
            mv = self.matvec(Regu,p)

            prod = self.tenpy.mult_lists(p,mv)

            alpha = self.tenpy.mult_lists(r,r)/prod

            x = self.tenpy.scl_list_add(alpha,x,p)

            r_new = self.tenpy.scl_list_add(-1*alpha,r,mv)
                
            #self.tenpy.printf('res in cg is',self.tenpy.list_vecnorm(r_new))

            if self.tenpy.list_vecnorm(r_new)<tol:
                counter+=1
                #end = time.time()
                break
            beta = self.tenpy.mult_lists(r_new,r_new)/self.tenpy.mult_lists(r,r)

            p = self.tenpy.scl_list_add(beta,r_new,p)
            r = r_new
            counter += 1

            if counter == self.maxiter:
                #end = time.time()
                break
                
        #self.tenpy.printf('cg took',end-start)
        

        return x,counter

    def fast_precond_conjugate_gradient(self,g,P,Regu):
        #start = time.time()
        
        x = [self.tenpy.zeros(A.shape) for A in g]
        
        g_norm = self.tenpy.list_vecnorm(g)
            

        tol = np.max([self.atol,np.min([self.cg_tol,np.sqrt(g_norm)])])*g_norm
        
        if g_norm<tol:
            return x

        z = fast_block_diag_precondition(self.tenpy,g,P)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu,p)

            mul = self.tenpy.mult_lists(g,z)

            alpha = mul/self.tenpy.mult_lists(p,mv) 

            x =self.tenpy.scl_list_add(alpha,x,p)

            g = self.tenpy.scl_list_add(-1*alpha,g,mv)
            
            
            if self.tenpy.list_vecnorm(g)<tol:
                counter+=1
                #end = time.time()
                break

            z = fast_block_diag_precondition(self.tenpy,g,P)

            beta = self.tenpy.mult_lists(g,z)/mul

            p = self.tenpy.scl_list_add(beta,z,p)

            counter += 1
            
            if counter == self.maxiter:
                #end = time.time()
                break
                
        #end = time.time()
        #self.tenpy.printf("cg took:",end-start)

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



    def update_A(self,delta,alpha):
        for i in range(len(delta)):
            self.A[i] += alpha*delta[i]

    def update_temp(self,delta,alpha):
        for i in range(len(delta)):
            self.temp[i] += alpha*delta[i]
            
    
    def armijo_line(self,A_res,delta,g,alpha=1.0):
        m = self.tenpy.mult_lists(g,delta)
        t= self.c*m
        for i in range(self.arm_iter):
            self.update_temp(delta,alpha)
            if get_residual(self.tenpy,self.T,self.temp) - A_res  <= alpha*t:
                break
            else:
                for i in range(len(self.temp)):
                    self.temp[i] = self.A[i].copy()
                alpha = self.tau*alpha

        return alpha

    def step(self,Regu):
        self.compute_G()
        self.compute_gamma()
        
        g= self.gradient()
        
        self.g_norm = self.tenpy.list_vecnorm(g)
        
        
        P = self.compute_block_diag_preconditioner(Regu)

        [self.delta,counter] = self.fast_precond_conjugate_gradient(g,P,Regu)
        self.total_iters+= counter
        
        
        self.atol = self.num*self.tenpy.list_vecnorm(self.delta)
        
        
        self.temp = np.zeros_like(self.A)
        
        for i in range(len(self.delta)):
            self.temp[i] = self.A[i].copy()
            
        if self.Arm:
            A_res = get_residual(self.tenpy,self.T,self.A)
            alpha = self.armijo_line(A_res,self.delta,g)
            self.update_A(self.delta,alpha)
            
        else:
            alpha = 1
            self.update_A(self.delta,alpha)
            
        
        #self.tenpy.printf("total cg iterations",self.total_iters)
        self.nls_iter+=1
        
        
        return [self.A,self.total_iters]



    def step2(self,Regu):
        
        def cg_call(v):
            self.total_iters+=1

        self.compute_G()
        self.compute_gamma()
        g = flatten_Tensor(self.tenpy,self.gradient())
        mult_LinOp = self.create_fast_hessian_contract_LinOp(Regu)
        P = self.compute_block_diag_preconditioner(Regu)
        precondition_LinOp = self.create_block_precondition_LinOp(P)
        [delta,_] = spsalg.cg(mult_LinOp,-1*g,tol=self.cg_tol,M=precondition_LinOp,callback=cg_call,atol=self.atol)
        #[delta,_] = spsalg.cg(mult_LinOp,-1*g,tol=self.cg_tol,callback=cg_call,atol=self.atol)
        self.atol = self.num*self.tenpy.norm(delta)
        delta = reshape_into_matrices(self.tenpy,delta,self.A)
        self.update_A(delta)
        self.tenpy.printf('total cg iterations:',self.total_iters)
        
        
        
        return [self.A,self.total_iters]


