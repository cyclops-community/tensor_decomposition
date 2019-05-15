import numpy as np
import sys
import time
import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join
import CPD.common_kernels as ck
import CPD.standard_ALS as stnd_ALS
import tensors.synthetic_tensors as stsrs
import argparse
import arg_defs as arg_defs
import csv

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def test_rand_naive(tenpy,A,T,O,num_iter,sp_res,csv_writer=None,Regu=None):
    time_all = 0.
    for i in range(num_iter):
        if sp_res:
            res = ck.get_residual_sp(tenpy,O,T,A)
        else:
            res = ck.get_residual(tenpy,T,A)
        if tenpy.is_master_proc():
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i, time_all, res
            ])
        t0 = time.time()
        A = stnd_ALS.dt_ALS_step(tenpy,T,A,Regu)
        t1 = time.time()
        tenpy.printf("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    tenpy.printf("Naive method took",time_all,"seconds overall")



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
    sp_frac = args.sp_fraction
    sp_res = args.sp_res
    tensor = args.tensor
    tlib = args.tlib

    if tlib == "numpy":
        import numpy_ext as tenpy
    elif tlib == "ctf":
        import ctf_ext as tenpy
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


    if tensor == "random":
        tenpy.printf("Testing random tensor")
        [T,O] = stsrs.init_rand(tenpy,order,s,R,sp_frac)
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
    for i in range(T.ndim):
        A.append(tenpy.random((T.shape[i],R)))
    test_rand_naive(tenpy,A,T,O,num_iter,sp_res,csv_writer,Regu)

