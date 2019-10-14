from CPD.common_kernels import compute_number_of_variables, compute_lin_sys, flatten_Tensor, reshape_into_matrices, solve_sys, get_residual
from scipy.sparse.linalg import LinearOperator
from CPD.standard_ALS import CP_DTALS_Optimizer

import scipy.sparse.linalg as spsalg
import numpy as np
import time

try:
    import Queue as queue
except ImportError:
    import queue


def fast_hessian_contract_batch(tenpy, XX, AA, GD, GG, regu=1):
    RR = regu * XX
    RR += tenpy.einsum("niz,nzr->nir", XX, GD)
    if tenpy.name() == 'numpy':
        RR += np.einsum("niz,pjr,npzr,pjz->nir", AA, AA, GG, XX, optimize=True)
    else:
        RR += tenpy.einsum("niz,pjr,npzr,pjz->nir", AA, AA, GG, XX)
    return RR


def fast_hessian_contract(tenpy, X, A, gamma, regu=1):
    N = len(A)
    ret = []
    for n in range(N):
        ret.append(tenpy.zeros(A[n].shape))
        for p in range(N):
            M = gamma[n][p]
            if n == p:
                ret[n] += tenpy.einsum("iz,zr->ir", X[p], M)
            else:
                B = tenpy.einsum("jr,jz->rz", A[p], X[p])
                ret[n] += tenpy.einsum("iz,zr,rz->ir", A[n], M, B)
        ret[n] += regu * X[n]

    return ret


def fast_block_diag_precondition(tenpy, X, P):
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
    def __init__(self,
                 tenpy,
                 T,
                 A,
                 maxiter,
                 cg_tol=1e-3,
                 num=0,
                 try_batch=False,
                 args=None):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.cg_tol = cg_tol
        self.num = num
        self.G = None
        self.gamma = None
        self.atol = 0
        self.total_iters = 0
        self.maxiter = maxiter
        self.nls_iter = 0
        self.g_norm = 0
        self.alpha = 1
        self.DTD = None
        self.GG = None
        self.GD = None
        s = A[0].shape[0]

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = "".join([chr(ord('a') + j) for j in range(nd)]) + ci
        str2 = (chr(ord('a') + ii)) + "R"
        str3 = "".join([chr(ord('a') + j) for j in range(nd) if j != ii]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def compute_G(self):
        G = []
        for i in range(len(self.A)):
            G.append(self.tenpy.einsum("ij,ik->jk", self.A[i], self.A[i]))
        self.G = G

    def compute_coefficient_matrix(self, n1, n2):
        ret = self.tenpy.ones(self.G[0].shape)
        for i in range(len(self.G)):
            if i != n1 and i != n2:
                ret = self.tenpy.einsum("ij,ij->ij", ret, self.G[i])
        return ret

    def compute_gamma(self):
        N = len(self.A)
        result = []
        for i in range(N):
            result.append([])
            for j in range(N):
                if j >= i:
                    M = self.compute_coefficient_matrix(i, j)
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
                if j >= i:
                    M = self.compute_coefficient_matrix(i, j)
                    result[i].append(M)
                else:
                    M = result[j][i]
                    result[i].append(M)
        return result

    def get_diagonal(self, g):
        dag = []
        for i in range(len(self.A)):
            dag.append(
                np.maximum(
                    np.repeat(self.gamma[i][i].diagonal(), self.A[i].shape[0]),
                    g[i].reshape(-1, order='F')))
        dag = np.array(dag)

        self.DTD = np.diag(dag.reshape(-1))

    def compute_block_diag_preconditioner(self, Regu):
        P = []
        for i in range(len(self.A)):
            n = self.gamma[i][i].shape[0]
            P.append(
                self.tenpy.cholesky(self.gamma[i][i] +
                                    Regu * self.tenpy.eye(n)))
        return P

    def gradient(self):
        grad = []
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))), self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert (len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0]) - 1
                if idx == len(s[-1][0]) - 1:
                    ii = len(s[-1][0]) - 2

                einstr = self._einstr_builder(M, s, ii)

                N = self.tenpy.einsum(einstr, M, self.A[ii])

                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss, N))
            M = s[-1][1]
            g = -1 * M + self.A[i].dot(self.gamma[i][i])
            grad.append(g)
        return grad

    def power_method(self, l, iter=1):
        for i in range(iter):
            l = fast_hessian_contract(self.tenpy, l, self.A, self.gamma, 0)
            a = self.tenpy.list_vecnorm(l)
            l = self.tenpy.scalar_mul(1 / a, l)
        return l

    def rayleigh_quotient(self, l):
        a = self.tenpy.mult_lists(
            l, fast_hessian_contract(self.tenpy, l, self.A, self.gamma, 0))
        b = self.tenpy.list_vecnormsq(l)
        return a / b

    def create_fast_hessian_contract_LinOp(self, Regu):
        num_var = compute_number_of_variables(self.A)
        A = self.A
        gamma = self.gamma
        tenpy = self.tenpy
        template = self.A

        def mv(delta):
            delta = reshape_into_matrices(tenpy, delta, template)
            result = fast_hessian_contract(tenpy, delta, A, gamma, Regu)
            vec = flatten_Tensor(tenpy, result)
            return vec

        V = LinearOperator(shape=(num_var, num_var), matvec=mv)
        return V

    def matvec(self, Regu, delta):
        t0 = time.time()
        A = self.A
        gamma = self.gamma
        tenpy = self.tenpy
        template = self.A
        result = fast_hessian_contract(tenpy, delta, A, gamma, Regu)
        t1 = time.time()
        return result

    def fast_conjugate_gradient(self, g, Regu):
        start = time.time()

        x = [self.tenpy.zeros(A.shape) for A in g]

        tol = np.max([self.atol, self.cg_tol * self.tenpy.list_vecnorm(g)])

        r = self.tenpy.list_add(
            self.tenpy.scalar_mul(-1, g),
            self.tenpy.scalar_mul(-1, self.matvec(Regu, x)))

        if self.tenpy.list_vecnorm(r) < tol:
            return x
        p = r
        counter = 0

        while True:
            mv = self.matvec(Regu, p)

            alpha = self.tenpy.list_vecnormsq(r) / self.tenpy.mult_lists(p, mv)

            x = self.tenpy.list_add(x, self.tenpy.scalar_mul(alpha, p))

            r_new = self.tenpy.list_add(
                r, self.tenpy.scalar_mul(-1, self.tenpy.scalar_mul(alpha, mv)))

            if self.tenpy.list_vecnorm(r_new) < tol:
                counter += 1
                end = time.time()
                break
            beta = self.tenpy.list_vecnormsq(
                r_new) / self.tenpy.list_vecnormsq(r)

            p = self.tenpy.list_add(r_new, self.tenpy.scalar_mul(beta, p))
            r = r_new
            counter += 1

            if counter == self.maxiter:
                end = time.time()
                break

        self.tenpy.printf('cg took', end - start)

        return x, counter

    def fast_precond_conjugate_gradient(self, g, P, Regu):
        start = time.time()

        x = [self.tenpy.zeros(A.shape) for A in g]

        tol = np.max([self.atol, self.cg_tol * self.tenpy.list_vecnorm(g)])

        r = self.tenpy.list_add(
            self.tenpy.scalar_mul(-1, g),
            self.tenpy.scalar_mul(-1, self.matvec(Regu, x)))

        if self.tenpy.list_vecnorm(r) < tol:
            return x

        z = fast_block_diag_precondition(self.tenpy, r, P)

        p = z

        counter = 0
        while True:
            mv = self.matvec(Regu, p)

            mul = self.tenpy.mult_lists(r, z)

            alpha = mul / self.tenpy.mult_lists(p, mv)

            x = self.tenpy.list_add(x, self.tenpy.scalar_mul(alpha, p))

            r_new = self.tenpy.list_add(
                r, self.tenpy.scalar_mul(-1, self.tenpy.scalar_mul(alpha, mv)))

            if self.tenpy.list_vecnorm(r_new) < tol:
                counter += 1
                end = time.time()
                break

            z_new = fast_block_diag_precondition(self.tenpy, r_new, P)
            beta = self.tenpy.mult_lists(r_new, z_new) / mul

            p = self.tenpy.list_add(z_new, self.tenpy.scalar_mul(beta, p))

            r = r_new
            z = z_new
            counter += 1

            if counter == self.maxiter:
                end = time.time()
                break

        end = time.time()
        self.tenpy.printf("cg took:", end - start)

        return x, counter

    def create_block_precondition_LinOp(self, P):
        num_var = compute_number_of_variables(self.A)
        tenpy = self.tenpy
        template = self.A

        def mv(delta):

            delta = reshape_into_matrices(tenpy, delta, template)
            result = fast_block_diag_precondition(tenpy, delta, P)
            vec = flatten_Tensor(tenpy, result)
            return vec

        V = LinearOperator(shape=(num_var, num_var), matvec=mv)
        return V

    def update_A(self, delta, alpha):
        for i in range(len(delta)):
            self.A[i] += alpha * delta[i]

    def step(self, Regu):
        self.compute_G()
        self.compute_gamma()

        g = self.gradient()
        self.g_norm = self.tenpy.list_vecnorm(g)

        self.tenpy.printf('gradient norm is', self.g_norm)

        P = self.compute_block_diag_preconditioner(Regu)

        [delta, counter] = self.fast_precond_conjugate_gradient(g, P, Regu)

        self.total_iters += counter

        self.atol = self.num * self.tenpy.list_vecnorm(delta)
        self.tenpy.printf('cg iterations:', counter)

        self.update_A(delta, self.alpha)

        self.tenpy.printf("total cg iterations", self.total_iters)
        self.nls_iter += 1

        return [self.A, self.total_iters]

    def step2(self, Regu):
        def cg_call(v):
            self.total_iters += 1

        self.compute_G()
        self.compute_gamma()
        g = flatten_Tensor(self.tenpy, self.gradient())
        mult_LinOp = self.create_fast_hessian_contract_LinOp(Regu)
        P = self.compute_block_diag_preconditioner(Regu)
        precondition_LinOp = self.create_block_precondition_LinOp(P)
        [delta, _] = spsalg.cg(mult_LinOp,
                               -1 * g,
                               tol=self.cg_tol,
                               M=precondition_LinOp,
                               callback=cg_call,
                               atol=self.atol)
        self.atol = self.num * self.tenpy.norm(delta)
        delta = reshape_into_matrices(self.tenpy, delta, self.A)
        self.update_A(delta)
        self.tenpy.printf('total cg iterations:', self.total_iters)

        return [self.A, self.total_iters]
