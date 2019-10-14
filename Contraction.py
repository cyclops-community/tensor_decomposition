from CPD.NLS import fast_hessian_contract, CP_fastNLS_Optimizer
from CPD.common_kernels import compute_number_of_variables, flatten_Tensor, reshape_into_matrices, solve_sys, get_residual
from CPD.standard_ALS import CP_DTALS_Optimizer
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tlib',
        default="ctf",
        metavar='string',
        choices=[
            'ctf',
            'numpy',
        ],
        help=
        'choose tensor library teo test, choose between numpy and ctf (default: ctf)'
    )

    args, _ = parser.parse_known_args()
    tlib = args.tlib

    if tlib == "numpy":
        import backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import backend.ctf_ext as tenpy
        import ctf

    X = []
    delta = []
    size = 500

    for i in range(3):
        X.append(tenpy.random((size, size)))
        delta.append(0.1 * tenpy.random((size, size)))

    T = tenpy.random((size, size, size))
    opt = CP_fastNLS_Optimizer(tenpy, T, X, 1, 1e-08, 0)

    t1 = time.time()
    start = time.time()
    opt.compute_G()
    opt.compute_gamma()
    g = opt.gradient()
    end = time.time()

    print('time taken for' + tlib + 'to compute gradient and preconditioner:',
          end - start)
    vals = opt.fast_conjugate_gradient(g, 1)

    t2 = time.time()

    print('time taken for NLS for' + tlib + 'is:', t2 - t1)

    t1 = time.time()
    opt2 = CP_DTALS_Optimizer(tenpy, T, X)
    opt2.step(0)
    t2 = time.time()

    print('time taken for ALS for' + tlib + 'is:', t2 - t1)
