import numpy as np
import ctf
from ctf import random
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

def test_rand_naive(A,T,O,num_iter,sp_res,csv_writer=None,Regu=None):
    time_all = 0.
    optimizer = stnd_ALS.DTALS_Optimizer()

    for i in range(num_iter):
        if sp_res:
            res = ck.get_residual_sp(O,T,A)
        else:
            res = ck.get_residual(T,A)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i, time_all, res
            ])
        t0 = time.time()
        A = optimizer.step(T,A,Regu)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    if ctf.comm().rank() == 0:
        print("Naive method took",time_all,"seconds overall")



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

    w = ctf.comm()

    if w.rank() == 0 :
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual'
            ])

    s = args.s
    order = args.order
    R = args.R
    r = args.r
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    sp_res = args.sp_res
    Regu = args.regularization * ctf.eye(R,R)
    tensor = args.tensor

    if tensor == "random":
        if w.rank() == 0 :
            print("Testing random tensor")
        [T,O] = stsrs.init_rand(order,s,R,sp_frac)
    elif tensor == "mom_cons":
        if w.rank() == 0 :
            print("Testing order 4 momentum conservation tensor")
        T = stsrs.init_mom_cons(s)
        O = None
        sp_res = False
    elif tensor == "mom_cons_sv":
        if w.rank() == 0 :
            print("Testing order 3 singular vectors of unfolding of momentum conservation tensor")
        T = stsrs.init_mom_cons_sv(s)
        O = None
        sp_res = False
    else:
        print("ERROR: Invalid --tensor input")

    A = []
    for i in range(T.ndim):
        A.append(ctf.random.random((T.shape[i],R)))
    test_rand_naive(A,T,O,num_iter,sp_res,csv_writer,Regu)

