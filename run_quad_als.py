import numpy as np
import numpy.linalg as la
import tensors.real_tensors as real_tensors
from CPD.common_kernels import get_residual
from os.path import dirname, join
import argparse
from pathlib import Path
import csv
import time

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


def get_file_prefix(args):
    return "-".join(filter(None, [
        args.experiment_prefix,
        'R' + str(args.R),
        args.method,
    ]))


class quad_als_optimizer():

    def __init__(self, tenpy, T, A, B):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.B = B

    def step(self):

        T_B = self.tenpy.einsum("ijk,ka->ija", self.T, self.B)
        M = self.tenpy.einsum("ija,ja->ai", T_B, self.B)
        BB = self.tenpy.einsum("ja,jb->ab", self.B, self.B)
        S = BB * BB
        self.A = la.solve(S, M).T
        # R = T - self.tenpy.einsum("ia,ja,ka->ijk", X, B, B)
        # print("Residual norm after X update is", la.norm(R))
        M = self.tenpy.einsum("ijk,ia->ajk", T, self.A)
        N = self.tenpy.einsum("ia,ib->ab", self.A, self.A)
        max_approx = 5
        for i in range(max_approx):
            lam = .2
            S = N * self.tenpy.einsum("ia,ib->ab", self.B, self.B)
            MM = self.tenpy.einsum("ajk,ka->aj", M, self.B)
            self.B = lam * self.B + (1 - lam) * la.solve(S, MM).T
            # R = self.T - self.tenpy.einsum("ia,ja,ka->ijk", self.A, self.B, self.B)
            # print("Residual norm after", max_approx, "approx step is", la.norm(R))

        return self.A, self.B


class quad_pp_optimizer(quad_als_optimizer):

    def __init__(self, tenpy, T, A, B, args):

        quad_als_optimizer.__init__(self, tenpy, T, A, B)
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt
        self.dA = tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.T_A0 = None
        self.T_B0 = None
        self.T_A0_B0 = None
        self.T_B0_B0 = None

    def _step_dt(self):
        return quad_als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        self.T_A0 = self.tenpy.einsum("ijk,ia->ajk", self.T, self.A)
        self.T_B0 = self.tenpy.einsum("ijk,ka->ija", self.T, self.B)
        self.T_A0_B0 = self.tenpy.einsum("ajk,ka->aj", self.T_A0, self.B)
        self.T_B0_B0 = self.tenpy.einsum("ija,ja->ai", self.T_B0, self.B)
        self.dA = tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = tenpy.zeros((self.B.shape[0], self.B.shape[1]))

    def _step_pp_subroutine(self):
        print("***** pairwise perturbation step *****")

        M = self.T_B0_B0.copy()
        M = M + 2 * self.tenpy.einsum("ija,ja->ai", self.T_B0, self.dB)
        BB = self.tenpy.einsum("ja,jb->ab", self.B, self.B)
        S = BB * BB
        A_new = la.solve(S, M).T

        self.dA = self.dA + A_new - self.A
        self.A = A_new

        N = self.tenpy.einsum("ia,ib->ab", self.A, self.A)
        max_approx = 5
        for i in range(max_approx):
            lam = .2
            S = N * self.tenpy.einsum("ia,ib->ab", self.B, self.B)
            M = self.T_A0_B0 + \
                self.tenpy.einsum("ajk,ka->aj", self.T_A0, self.dB)
            M = M + self.tenpy.einsum("ija,ia->aj", self.T_B0, self.dA)
            # M = M + self.tenpy.einsum("ijk,ia,ka->aj", self.T, self.dA, self.dB)
            B_new = lam * self.B + (1 - lam) * la.solve(S, M).T
            self.dB = self.dB + B_new - self.B
            self.B = B_new

        smallupdates = True
        norm_dA = self.tenpy.sum(self.dA**2)**.5
        norm_dB = self.tenpy.sum(self.dB**2)**.5
        norm_A = self.tenpy.sum(self.A**2)**.5
        norm_B = self.tenpy.sum(self.B**2)**.5
        if norm_dA > self.tol_restart_dt * norm_A or self.tenpy.sum(
                self.dB**2)**.5 > self.tol_restart_dt * norm_B:
            smallupdates = False

        if smallupdates is False:
            self.pp = False
            self.reinitialize_tree = False

        return self.A, self.B

    def _step_dt_subroutine(self):
        A_prev, B_prev = self.A.copy(), self.B.copy()
        self._step_dt()
        smallupdates = True

        self.dA = self.A - A_prev
        self.dB = self.B - B_prev
        norm_dA = self.tenpy.sum(self.dA**2)**.5
        norm_dB = self.tenpy.sum(self.dB**2)**.5
        norm_A = self.tenpy.sum(self.A**2)**.5
        norm_B = self.tenpy.sum(self.B**2)**.5
        # print(norm_dA, norm_dB)
        if norm_dA >= self.tol_restart_dt * norm_A or norm_dB >= self.tol_restart_dt * norm_B:
            smallupdates = False

        if smallupdates is True:
            self.pp = True
            self.reinitialize_tree = True
        return self.A, self.B

    def step(self):
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                restart = True
                self._initialize_tree()
                self.reinitialize_tree = False
            A, B = self._step_pp_subroutine()
        else:
            A, B = self._step_dt_subroutine()
        return A, B, restart


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-prefix',
        '-ep',
        type=str,
        default='',
        required=False,
        metavar='str',
        help='Output csv file name prefix (default: None)')
    parser.add_argument(
        '--R',
        type=int,
        default=1000,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--method',
        default="DT",
        metavar='string',
        choices=[
            'DT',
            'PP',
        ],
        help='choose the optimization method: DT, PP (default: DT)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='int',
        help='random seed')
    parser.add_argument(
        '--tol-restart-dt',
        default=1.,
        type=float,
        metavar='float',
        help='used in pairwise perturbation optimizer, tolerance for dimention tree restart')

    args, _ = parser.parse_known_args()

    import backend.numpy_ext as tenpy

    # TODO: currently all the methods are messed up. Needs to refactor a lot.
    flag_dt = True

    R = args.R
    res_calc_freq = 10

    csv_path = join(results_dir, get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')  # , newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args):
            print(arg + ':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual', 'fitness', ''
            ])

    tenpy.seed(args.seed)

    T = real_tensors.get_scf_tensor(tenpy)
    tenpy.printf("The shape of the input tensor is: ", T.shape)

    X = tenpy.random((T.shape[0], R))
    Y = tenpy.random((T.shape[1], R))

    optimizer_list = {
        'DT': quad_als_optimizer(tenpy, T, X, Y),
        'PP': quad_pp_optimizer(tenpy, T, X, Y, args),
    }
    optimizer = optimizer_list[args.method]

    normT = tenpy.vecnorm(T)
    time_all = 0.

    for i in range(2000):
        if i % res_calc_freq == 0 or i == 2000 - 1 or not flag_dt:
            res = get_residual(tenpy, T, [X, Y, Y])
            fitness = 1 - res / normT
            if tenpy.is_master_proc():
                print("[", i, "] Residual is", res, "fitness is: ", fitness)
                if csv_file is not None:
                    csv_writer.writerow([i, time_all, res, fitness, flag_dt])
                    csv_file.flush()
        t0 = time.time()
        if args.method == 'PP':
            X, Y, pp_restart = optimizer.step()
            flag_dt = not pp_restart
        else:
            X, Y = optimizer.step()
        t1 = time.time()
        tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")
        time_all += t1 - t0
