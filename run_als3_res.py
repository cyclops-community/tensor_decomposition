import numpy as np
import sys
import time
import argparse
import csv
from pathlib import Path
from os.path import dirname, join
import tensors.synthetic_tensors as synthetic_tensors
from CPD.common_kernels import get_residual
from CPD.common_kernels import solve_sys, compute_lin_sys

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

class cp_dtals_res_optimizer():

    def __init__(self, tenpy, T, A):
        self.tenpy = tenpy
        self.T = T
        self.A = A

    def step(self, Regu):
        # 1st mode
        res_T = self.T - self.tenpy.einsum("ia,ja,ka->ijk", self.A[0], self.A[1], self.A[2])
        TC = self.tenpy.einsum("ijk,ka->ija", res_T, self.A[2])
        rhs = self.tenpy.einsum("ija,ja->ia", TC, self.A[1])
        self.A[0] += solve_sys(self.tenpy, compute_lin_sys(self.tenpy,
                                                      self.A[1], self.A[2], Regu), rhs)
        # 2nd mode
        res_T = self.T - self.tenpy.einsum("ia,ja,ka->ijk", self.A[0], self.A[1], self.A[2])
        TC = self.tenpy.einsum("ijk,ka->ija", res_T, self.A[2])
        rhs = self.tenpy.einsum("ija,ia->ja", TC, self.A[0])
        self.A[1] += solve_sys(self.tenpy, compute_lin_sys(self.tenpy,
                                                      self.A[0], self.A[2], Regu), rhs)
        # 3rd mode
        res_T = self.T - self.tenpy.einsum("ia,ja,ka->ijk", self.A[0], self.A[1], self.A[2])
        rhs = self.tenpy.einsum("ijk,ia,ja->ka", res_T, self.A[0], self.A[1])
        self.A[2] += solve_sys(self.tenpy, compute_lin_sys(self.tenpy,
                                                      self.A[0], self.A[1], Regu), rhs)
        return self.A


def CP_ALS_res(
        tenpy,
        A,
        T,
        O,
        num_iter,
        csv_file=None,
        Regu=None,
        method='DT_res',
        res_calc_freq=1,
        tol=1e-05):

    assert(T.ndim == 3)

    from CPD.standard_ALS import CP_DTALS_Optimizer

    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    if Regu is None:
        Regu = 0
    
    
    normT = tenpy.vecnorm(T)

    optimizer_list = {
        'DT': CP_DTALS_Optimizer(tenpy, T, A),
        'DT_res': cp_dtals_res_optimizer(tenpy, T, A),
    }
    optimizer = optimizer_list[method]

    time_all = 0.
    
    decrease= True
    increase=False
    
    flag = False

    for i in range(num_iter):

        if i % res_calc_freq == 0 or i == num_iter - 1:
            res = get_residual(tenpy, T, A)
            fitness = 1 - res / normT

            if tenpy.is_master_proc():
                print("[", i, "] Residual is", res, "fitness is: ", fitness)
                # write to csv file
                csv_writer.writerow([i, time_all, res, fitness])
                csv_file.flush()

        if res < tol:
            print('Method converged in', i, 'iterations')
            break

        t0 = time.time()
        # Regu = 1/(i+1)
        print("Regu is:", Regu)

        A = optimizer.step(Regu)
        
        
        #if res < 1:
        #    flag = True
            
        #if flag:
        #    Regu = 1e-05
            
        #else:
        if Regu < 1e-05:
            increase=True
            decrease=False
        
        if Regu > 1e-01:
            decrease= True
            increase=False
                
        
            
        if increase:
            Regu = Regu*2
            
        elif decrease:
            Regu = Regu/2

        t1 = time.time()
        tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")

        time_all += t1 - t0

    tenpy.printf(method + " method took", time_all, "seconds overall")

    return res


def get_file_prefix(args):
    return "-".join(filter(None, [
        args.experiment_prefix,
        'R' + str(args.R),
        args.method,
        'Regu' + str(args.regularization),
    ]))


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
        default=10,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--s',
        type=int,
        default=1000,
        metavar='int',
        help='Input CP decomposition size (default: 60)')
    parser.add_argument(
        '--method',
        default="DT_res",
        metavar='string',
        choices=[
            'DT',
            'DT_res',
        ],
        help='choose the optimization method: DT, DT_res (default: DT_res)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='int',
        help='random seed')
    parser.add_argument(
        '--num-iter',
        type=int,
        default=200,
        metavar='int',
        help='Number of iterations (default: 200)')
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.0000001,
        metavar='float',
        help='regularization (default: 0.0000001)')
    parser.add_argument(
        '--tensor',
        default="random",
        metavar='string',
        choices=[
            'random',
            'random_col',
            'scf',
        ],
        help='choose tensor to test, available: random, random_col, scf (default: random)')
    parser.add_argument(
        '--col',
        type=float,
        nargs='+',
        default=[0.2, 0.8],
        help='collinearity range')

    args, _ = parser.parse_known_args()

    import backend.numpy_ext as tenpy

    s = args.s
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
                'iterations', 'time', 'residual', 'fitness',
            ])

    tenpy.seed(args.seed)

    if args.tensor == "random":
        tenpy.printf("Testing random tensor")
        [T, O] = synthetic_tensors.init_rand(tenpy, 3, s, R, 1., args.seed)
    elif args.tensor == "random_col":
        [T, O] = synthetic_tensors.init_collinearity_tensor(
            tenpy, s, 3, R, args.col, args.seed)
    elif args.tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)
        O = None

    tenpy.printf("The shape of the input tensor is: ", T.shape)

    Regu = args.regularization

    A = []
    for i in range(T.ndim):
        A.append(tenpy.random((s, R)))

    CP_ALS_res(
        tenpy,
        A,
        T,
        O,
        args.num_iter,
        csv_file,
        Regu,
        args.method,
        res_calc_freq,
        tol=1e-05)
