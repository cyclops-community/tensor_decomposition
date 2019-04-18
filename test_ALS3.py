import numpy as np
import ctf
from ctf import random
import sys
import time
import common_ALS3_kernels as cak
import lowr_ALS3 as lowr_ALS
import standard_ALS3 as stnd_ALS
import sliced_ALS3 as slic_ALS
import synthetic_tensors as stsrs

import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def add_general_arguments(parser):

    parser.add_argument(
        '--experiment-prefix',
        '-ep',
        type=str,
        default='',
        required=False,
        metavar='str',
        help='Output csv file name prefix (default: None)')
    parser.add_argument(
        '--s',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in each dimension (default: 64)')
    parser.add_argument(
        '--R',
        type=int,
        default=10,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--r',
        type=int,
        default=10,
        metavar='int',
        help='Update rank size (default: 10)')
    parser.add_argument(
        '--num-iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations (default: 10)')
    parser.add_argument(
        '--num-lowr-init-iter',
        type=int,
        default=2,
        metavar='int',
        help='Number of initializing iterations (default: 2)')
    parser.add_argument(
        '--sp-fraction',
        type=float,
        default=1.,
        metavar='float',
        help='sparsity (default: 1)')
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.1,
        metavar='float',
        help='regularization (default: 0.1)')
    parser.add_argument(
        '--sp-updatelowrank',
        type=int,
        default=0,
        metavar='int',
        help='mem-preserving ordering of low-rank sparse contractions (default: 0)')
    parser.add_argument(
        '--sp-res',
        type=int,
        default=0,
        metavar='int',
        help='TTTP-based sparse residual calculation (default: 0)')
    parser.add_argument(
        '--run-naive',
        type=int,
        default=1,
        metavar='int',
        help='Run naive Dimension tree algorithm (default: 1)')
    parser.add_argument(
        '--run-lowrank',
        type=int,
        default=0,
        metavar='int',
        help='Run Dimension tree algorithm with low rank update (default: 0)')
    parser.add_argument(
        '--mm-test',
        type=int,
        default=0,
        metavar='int',
        help='decompose matrix multiplication tensor as opposed to random (default: 0)')
    parser.add_argument(
        '--pois-test',
        type=int,
        default=0,
        metavar='int',
        help='decompose Poisson tensor as opposed to random (default: 0)')
    parser.add_argument(
        '--num-slices',
        type=int,
        default=1,
        metavar='int',
        help='if greater than one do sliced standard ALS with this many slices (default: 1)')

def get_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix, 
            's' + str(args.s),
            'R' + str(args.R),
            'r' + str(args.r),
            'spfrac' + str(args.sp_fraction),
            'splowrank' + str(args.sp_updatelowrank),
            'runlowrank' + str(args.run_lowrank),
            'pois' + str(args.pois_test),
            'numslices' + str(args.num_slices),
            'numinit-iter' + str(args.num_lowr_init_iter),
            'regu' + str(args.regularization),
        ]))

def test_rand_naive(s,R,num_iter,sp_frac,sp_res,mm_test=False,pois_test=False,csv_writer=None,Regu=None):
    if mm_test == True:
        [A,B,C,T,O] = stsrs.init_mm(s,R)
    elif pois_test == True:
        [A,B,C,T,O] = stsrs.init_poisson(s,R)
    else:
        [A,B,C,T,O] = stsrs.init_rand(s,R,sp_frac)
    time_all = 0.
    for i in range(num_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i, time_all, res
            ])
        t0 = time.time()
        [A,B,C] = stnd_ALS.dt_ALS_step(T,A,B,C,Regu)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    if ctf.comm().rank() == 0:
        print("Naive method took",time_all,"seconds overall")

def test_rand_sliced(s,R,num_iter,sp_frac,sp_res,num_slices,mm_test=False,pois_test=False,csv_writer=None,Regu=None):
    if mm_test == True:
        [A,B,C,T,O] = stsrs.init_mm(s,R)
    elif pois_test == True:
        [A,B,C,T,O] = stsrs.init_poisson(s,R)
    else:
        [A,B,C,T,O] = stsrs.init_rand(s,R,sp_frac)
    time_all = 0.
    Ta = []
    Tb = []
    Tc = []

    b = int((s+num_slices-1)/num_slices)
    for i in range(num_slices):
        st = i*b
        end = min(st+b,s)
        Ta.append(T[st:end,:,:])
        Tb.append(T[:,st:end,:])
        Tc.append(T[:,:,st:end])
        
    for i in range(num_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i, time_all, res
            ])
        t0 = time.time()
        [A,B,C] = slic_ALS.sliced_ALS_step(Ta,Tb,Tc,A,B,C,Regu)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Sweep took", t1-t0,"seconds")
        time_all += t1-t0
    if ctf.comm().rank() == 0:
        print("Naive method took",time_all,"seconds overall")


def test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul=False,sp_res=False,mm_test=False,pois_test=False,csv_writer=None,Regu=None):
    if mm_test == True:
        [A,B,C,T,O] = stsrs.init_mm(s,R)
    elif pois_test == True:
        [A,B,C,T,O] = stsrs.init_poisson(s,R)
    else:
        [A,B,C,T,O] = stsrs.init_rand(s,R,sp_frac)
    time_init = 0.

    for i in range(num_lowr_init_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i, time_init, res
            ])
        t0 = time.time()
        [A,B,C] = stnd_ALS.dt_ALS_step(T,A,B,C,Regu)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Full-rank sweep took", t1-t0,"seconds, iteration ",i)
        time_init += t1-t0
        if ctf.comm().rank() == 0:
            print ("Total time is ", time_init)
    time_lowr = time.time()

    if num_lowr_init_iter == 0:
        if ctf.comm().rank() == 0:
            print("Initializing leaves from low rank factor matrices")
        A1 = ctf.random.random((s,r))
        B1 = ctf.random.random((s,r))
        C1 = ctf.random.random((s,r))
        A2 = ctf.random.random((r,R))
        B2 = ctf.random.random((r,R))
        C2 = ctf.random.random((r,R))
        [RHS_A,RHS_B,RHS_C] = lowr_ALS.build_leaves_lowr(T,A1,A2,B1,B2,C1,C2)
        A = ctf.dot(A1,A2)
        B = ctf.dot(B1,B2)
        C = ctf.dot(C1,C2)
        if ctf.comm().rank() == 0:
            print("Done initializing leaves from low rank factor matrices")
    else:
        [RHS_A,RHS_B,RHS_C] = lowr_ALS.build_leaves(T,A,B,C)

    time_lowr = time.time() - time_lowr
    for i in range(num_iter-num_lowr_init_iter):
        if sp_res:
            res = cak.get_residual_sp(O,T,A,B,C)
        else:
            res = cak.get_residual(T,A,B,C)
        if ctf.comm().rank() == 0:
            print("Residual is", res)
            # write to csv file
            csv_writer.writerow([
                i+num_lowr_init_iter, time_init+time_lowr, res
            ])
        t0 = time.time()
        if sp_ul:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_ALS.lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,Regu,"update_leaves_sp")
        else:
            [A,B,C,RHS_A,RHS_B,RHS_C] = lowr_ALS.lowr_msdt_step(T,A,B,C,RHS_A,RHS_B,RHS_C,r,Regu)
        t1 = time.time()
        if ctf.comm().rank() == 0:
            print("Low-rank sweep took", t1-t0,"seconds, Iteration",i)
        time_lowr += t1-t0
        if ctf.comm().rank() == 0:
            print("Total time is ", time_lowr)
    if ctf.comm().rank() == 0:
        print("Low rank method (sparse update leaves =",sp_ul,") took",time_init,"for initial full rank steps",time_lowr,"for low rank steps and",time_init+time_lowr,"seconds overall")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up CSV logging
    csv_path = join(results_dir, get_file_prefix(args)+'.csv')
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
    R = args.R
    r = args.r
    num_iter = args.num_iter
    num_lowr_init_iter = args.num_lowr_init_iter
    sp_frac = args.sp_fraction
    sp_ul = args.sp_updatelowrank
    sp_res = args.sp_res
    run_naive = args.run_naive
    run_lowr = args.run_lowrank
    mm_test = args.mm_test
    num_slices = args.num_slices
    pois_test = args.pois_test

    Regu = args.regularization * ctf.eye(R,R)

    if run_naive:
        if num_slices == 1:
            if ctf.comm().rank() == 0:
                print("Testing naive version, printing residual before every ALS sweep")
            test_rand_naive(s,R,num_iter,sp_frac,sp_res,mm_test,pois_test,csv_writer,Regu)
        else:
            if ctf.comm().rank() == 0:
                print("Testing sliced version, printing residual before every ALS sweep")
            test_rand_sliced(s,R,num_iter,sp_frac,sp_res,num_slices,mm_test,pois_test,csv_writer,Regu)
    if run_lowr:
        if ctf.comm().rank() == 0:
            print("Testing low rank version, printing residual before every ALS sweep")
        test_rand_lowr(s,R,r,num_iter,num_lowr_init_iter,sp_frac,sp_ul,sp_res,mm_test,pois_test,csv_writer,Regu)
