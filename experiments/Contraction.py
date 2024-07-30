from tensor_decomposition.CPD.NLS import fast_hessian_contract, CP_fastNLS_Optimizer
from tensor_decomposition.CPD.common_kernels import compute_number_of_variables, flatten_Tensor, reshape_into_matrices, solve_sys, get_residual
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer
import argparse
import time
import numpy as np
import sys
import os
import csv
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
from pathlib import Path
from os.path import dirname, join

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tlib',
        default="numpy",
        metavar='string',
        choices=[
            'ctf',
            'numpy',
            ],
        help='choose tensor library to test, choose between numpy and numpy (default: numpy)')
    parser.add_argument(
        '--s',
        type=int,
        default=300,
        metavar="int",
        help="size of the tensor (s=R=size) for testing contractions, default is 300")
    parser.add_argument(
        '--R',
        type=int,
        default=300,
        metavar="int",
        help="Rank of the tensor (s=R=size) for testing contractions, default is 300")
    parser.add_argument(
    '--iterations',
    type=int,
    default=10,
    metavar="int",
    help="number of iterations")
    parser.add_argument(
    '--nodes',
    type=int,
    default=4,
    metavar="int",
    help="Number of nodes, default is 4")
    parser.add_argument(
    '--order',
    type=int,
    default=3,
    metavar="int",
    help="order of the tensor, default is 3")
    parser.add_argument(
    '--precond',
    type=int,
    default=1,
    metavar="int",
    help="If preconditioned iteration, choose 0 or 1, default is 1")
    
    
    
    args, _ = parser.parse_known_args()
    
    
    tlib = args.tlib
    s= args.s
    R = args.R
    nodes=args.nodes
    iterations = args.iterations
    order = args.order
    precond = args.precond
    
    csv_path = join(results_dir, 'new_svd_precond_test_batch_inc'+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        
        
    # initialize the csv file
    if tenpy.is_master_proc():
        if is_new_log:
            csv_writer.writerow([
                'precond','nodes','s','R','iterations','cg_median', 'mean_cg', 'std_cg', 'nls_median', 'mean_nls', 'std_nls','mean_start_up','std_start_up','mean_cg_batch','median_als', 'mean_als','std_als' 
            ])
        
  #  tenpy.printf('testing on',nodes,'nodes')
    
    X = []
    
    delta = []
    
    time_cg = []
    time_nls = []
    start_up = []
    
    time_cg_b = []
    time_nls_b = []
    start_up_b = []
    
    #tenpy.printf('performing warm up iteration')
    
    for i in range(order):
        X.append(tenpy.random((s,R)))
        delta.append(tenpy.random((s,R)))
    
    T = tenpy.random(order*[s])

    
    maxiter = 1
    cg_tol = 1e-08
    num = 0
    diag = 0
    Arm = 0
    c = 0 
    tau = 0 
    arm_iters = 0
    
    
    
    opt = CP_fastNLS_Optimizer(tenpy,T,X,maxiter,cg_tol,num,diag,Arm,c,tau,arm_iters,args)
    
    t1 = time.time()
    
    start = time.time()
    opt.compute_G()
    opt.compute_gamma()
    
    
    g= opt.gradient()
    
    if precond:
        P = opt.compute_block_diag_preconditioner(1)

    end = time.time()
    
    
    
    if precond:
        vals = opt.fast_precond_conjugate_gradient(g,P,1)
    else:
        vals = opt.fast_conjugate_gradient(g,1)
    #vals = opt.fast_conjugate_gradient_batch(g,1)
    
    t2= time.time()
    
    
    
    #tenpy.printf('warm up iteration completed')

    start1 = time.time()
    opt.compute_G()
    opt.compute_gamma()
    
    
    g= opt.gradient()
    #
    if precond:
        P = opt.compute_block_diag_preconditioner(1)

    end1 = time.time()
    
    
    
    for i in range(iterations):
        t1 = time.time()
        
        start = time.time()
        
        if precond:
            vals = opt.fast_precond_conjugate_gradient(g,P,1)
        
        else:
        
        #vals = opt.fast_conjugate_gradient_batch(g,1)
            vals = opt.fast_conjugate_gradient(g,1)
        
        end=time.time()
        
        time_cg+=[end-start]
        
        t2= time.time()
        
        
        time_nls+=[t2-t1]
        
    
  #  print('nls completed, moving to batch nls')
    gg = opt.gradient_GG(g)
    
    for i in range(iterations):
        
        start = time.time()
        
        
        #vals = opt.fast_conjugate_gradient_batch(g,1)
        vals = opt.fast_conjugate_gradient_batch(gg,1)
        
        end=time.time()
        
        time_cg_b+=[end-start]
        
        t2= time.time()
        
        
        time_nls_b+=[t2-t1]
        
    
    #print('batch nls completed, moving to als')
    
    
    
    
    opt2 = CP_DTALS_Optimizer(tenpy,T,X)
    
    opt2.step(1e-08)
    
    #print('warm up of als completed')
    
    time_als = []
    
    for i in range(iterations):
        
        
        t1 = time.time()
        
        #vals=opt2.step(1e-08)
            
        t2 = time.time()
        
        time_als+=[t2-t1]
        
        
    #print('\n time taken for cg batch steps is:',time_cg_b)
    
    #print('\n time taken for nls_batch is:',time_nls_b)
    
    
    #print('\n time taken for als is:',time_als)
    
    mean_cg = np.mean(time_cg)
    mean_nls = np.mean(time_nls)
    
    mean_cg_b = np.mean(time_cg_b)
    mean_nls_b = np.mean(time_nls_b)
    
    mean_als= np.mean(time_als)
    mean_start_up=  0
    
    #print('\n mean time taken for cg:',mean_cg)
    
    #print('\n mean time taken for nls:',mean_nls)
    
   # print('\n mean time taken for als:',mean_als)
    
    
    std_cg = np.std(time_cg)
    std_nls = np.std(time_nls)
    std_als = np.std(time_als)
    std_start_up = 0
    
    std_cg_b = np.std(time_cg_b)
    std_nls_b = np.std(time_nls_b)
       
       
    median_cg = np.median(time_cg)
    median_nls = np.median(time_nls)
    median_als = np.median(time_als)
    

    if tenpy.is_master_proc():
        if csv_file is not None:
            csv_writer.writerow([precond,nodes,s,R,iterations, median_cg,mean_cg, std_cg, median_nls, mean_nls, std_nls, mean_start_up, std_start_up,mean_cg_b, median_als, mean_als,std_als ])
            csv_file.flush()
            
    
