import numpy as np
import sys
import time
import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join
import tensors.synthetic_tensors as stsrs
import argparse
import arg_defs as arg_defs
import csv

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def CP_ALS(tenpy,A,T,O,r,num_iter,num_lowr_init_iter,sp_res,csv_writer=None,Regu=None,method='DT',tol_restart_dt=0.01):

    from CPD.common_kernels import get_residual_sp, get_residual
    from CPD.standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer
    from CPD.lowr_ALS import CP_DTLRALS_Optimizer

    time_all = 0.
    optimizer_list = {
        'DT': CP_DTALS_Optimizer(tenpy,T,A),
        'DTLR': CP_DTLRALS_Optimizer(tenpy,T,A,r),
        'PP': CP_PPALS_Optimizer(tenpy,T,A,tol_restart_dt),
    }

    if method == "DTLR":
        optimizer = optimizer_list['DT']
        for i in range(num_lowr_init_iter):
            if sp_res:
                res = get_residual_sp(tenpy,O,T,A)
            else:
                res = get_residual(tenpy,T,A)
            if tenpy.is_master_proc():
                print("[",i,"] Residual is", res)
                # write to csv file
                if csv_writer is not None:
                    csv_writer.writerow([ i, time_all, res ])
            t0 = time.time()
            A = optimizer.step(Regu)
            t1 = time.time()
            tenpy.printf("Init Sweep took", t1-t0,"seconds")
            time_all += t1-t0

    optimizer = optimizer_list[method]

    total_iter = num_iter - (method=='DTLR') * num_lowr_init_iter
    for i in range(total_iter):
        if sp_res:
            res = get_residual_sp(tenpy,O,T,A)
        else:
            res = get_residual(tenpy,T,A)
        if tenpy.is_master_proc():
            if method == 'DTLR':
                print("[",i+num_lowr_init_iter,"] Residual is", res)
            else:
                print("[",i,"] Residual is", res)
            # write to csv file
            if csv_writer is not None:
                csv_writer.writerow([ i, time_all, res ])
        t0 = time.time()
        A = optimizer.step(Regu)
        t1 = time.time()
        tenpy.printf("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    tenpy.printf(method+" method took",time_all,"seconds overall")

    return res

def Tucker_ALS(tenpy,A,T,O,num_iter,sp_res,csv_writer=None,Regu=None,method='DT',tol_restart_dt=0.01):

    from Tucker.common_kernels import get_residual_sp, get_residual
    from Tucker.standard_ALS import Tucker_DTALS_Optimizer, Tucker_PPALS_Optimizer

    time_all = 0.
    optimizer_list = {
        'DT': Tucker_DTALS_Optimizer(tenpy,T,A),
        'PP': Tucker_PPALS_Optimizer(tenpy,T,A,tol_restart_dt),
    }
    optimizer = optimizer_list[method]

    for i in range(num_iter):
        if sp_res:
            # TODO: implement the get residual sparse version
            res = get_residual_sp(tenpy,O,T,A)
        else:
            res = get_residual(tenpy,T,A)
        if tenpy.is_master_proc():
            print("[",i,"] Residual is", res)
            # write to csv file
            if csv_writer is not None:
                csv_writer.writerow([ i, time_all, res ])
        t0 = time.time()
        A = optimizer.step(Regu)
        t1 = time.time()
        tenpy.printf("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    tenpy.printf("Naive method took",time_all,"seconds overall")

    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up CSV logging
    csv_path = join(results_dir, arg_defs.get_file_prefix(args)+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    s = args.s
    order = args.order
    R = args.R
    r = args.r
    num_iter = args.num_iter
    num_lowr_init_iter = args.num_lowr_init_iter
    sp_frac = args.sp_fraction
    sp_res = args.sp_res
    tensor = args.tensor
    tlib = args.tlib

    if tlib == "numpy":
        import backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import backend.ctf_ext as tenpy
    else:
        print("ERROR: Invalid --tlib input")

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual'
            ])

    tenpy.seed(1)

    if tensor == "random":
        if args.decomposition == "CP":
            tenpy.printf("Testing random tensor")
            [T,O] = stsrs.init_rand(tenpy,order,s,R,sp_frac)
        if args.decomposition == "Tucker":
            tenpy.printf("Testing random tensor")
            shape = s * np.ones(order).astype(int)
            T = tenpy.random(shape)
            O = None
    elif tensor == "mom_cons":
        if tenpy.is_master_proc():
            print("Testing order 4 momentum conservation tensor")
        T = stsrs.init_mom_cons(tenpy,s)
        O = None
        sp_res = False
    elif tensor == "mom_cons_sv":
        tenpy.printf("Testing order 3 singular vectors of unfolding of momentum conservation tensor")
        T = stsrs.init_mom_cons_sv(tenpy,s)
        O = None
        sp_res = False
    else:
        print("ERROR: Invalid --tensor input")

    Regu = args.regularization * tenpy.eye(R,R)

    A = []
    if args.hosvd != 0:
        from Tucker.common_kernels import hosvd
        A = hosvd(tenpy, T, R, compute_core=False)
    else:
        for i in range(T.ndim):
            A.append(tenpy.random((T.shape[i],R)))

    if args.decomposition == "CP":
        CP_ALS(tenpy,A,T,O,r,num_iter,num_lowr_init_iter,sp_res,csv_writer,Regu,args.method,args.tol_restart_dt)
    elif args.decomposition == "Tucker":
        Tucker_ALS(tenpy,A,T,O,num_iter,sp_res,csv_writer,Regu,args.method,args.tol_restart_dt)
