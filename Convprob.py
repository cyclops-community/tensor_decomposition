import numpy as np
import time
import CPD.standard_ALS3 as stnd_ALS
import CPD.common_kernels as ck
import sys
import os
import argparse
from pathlib import Path
from os.path import dirname, join
import argparse
import arg_defs as arg_defs
import csv
from CPD.standard_ALS import CP_DTALS_Optimizer
from CPD.NLS import CP_fastNLS_Optimizer
from CPD.common_kernels import get_residual_sp, get_residual

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def convprob(tenpy,tensor,s,f_R,l_R,num_iter,num_gen,method,csv_writer=None,num_init = 10,Regu= 0.01,args=None):
    
    conv = []
    orig_Regu = Regu
    
    decrease= True
    increase=False
    lower = args.lower
    upper = args.upper
    varying = args.varying
    varying_fact = args.varying_fact
    conv_tol = args.conv_tol
    for R in range(f_R,l_R+1):

        for k in range(num_gen):
            if tensor == 'negrandom':
                a = np.random.uniform(low = -1, high = 1, size=  (s,R))
                b = np.random.uniform(low = -1, high = 1, size=  (s,R))
                c = np.random.uniform(low = -1, high = 1, size=  (s,R))
            
            elif tensor == 'randn':
                a = np.random.normal(size = (s,R))
                b = np.random.normal(size = (s,R))
                c = np.random.normal(size = (s,R))
            
            else:
                a = tenpy.random((s,R)) 
                b = tenpy.random((s,R))
                c = tenpy.random((s,R))

            T = tenpy.einsum('ia,ja,ka->ijk', a,b,c)
            converged = 0
            for j in range(num_init):
                total_iters = 0
                t_all = 0.0
                
                A = tenpy.random((s,R))
                P = A.copy()

                B = tenpy.random((s,R))
                Q = B.copy()

                C = tenpy.random((s,R))
                N = C.copy()

                X = [A,B,C]
                
                Regu = orig_Regu
                args.maxiter = 4*s*R
                optimizer_list = {
                'DT': CP_DTALS_Optimizer(tenpy,T,X,args),
                'NLS': CP_fastNLS_Optimizer(tenpy,T,X,args)}
                
                optimizer = optimizer_list[method]

                prev_res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                #print('Residual is',prev_res)
                
                start = time.time()
                for i in range(num_iter):
                    if method == 'NLS':
                        [X,iters] = optimizer.step(Regu)
                        
                    else:
                        delta = optimizer.step(Regu)
                 
                    
                    res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                    
                    
                    
                    if method == "NLS" :
                        if varying:
                            if Regu < lower:
                                increase=True
                                decrease=False
                    
                            if Regu > upper:
                                decrease= True
                                increase=False
                            
                            if increase:
                                Regu = Regu*varying_fact
                
                            elif decrease:
                                Regu = Regu/varying_fact
                    
                    if method == 'DT':
                        if varying:
                            if i%100 == 0:
                                lower = lower/2
                                upper = upper/2
                            if Regu < lower:
                                increase=True
                                decrease=False
                    
                            if Regu > upper:
                                decrease= True
                                increase=False
                            
                            if increase:
                                Regu = Regu*varying_fact
                
                            elif decrease:
                                Regu = Regu/varying_fact
                            
                    if res< conv_tol:
                        converged= 1
                        break
                    
                    if abs(prev_res - res)< 1e-07:
                        break
                    if method == 'NLS':
                        if optimizer.g_norm < 1e-15:
                            tenpy.printf('Method converged due to gradient tolerance in',i,'iterations')
                            break
                        if optimizer.g_norm > 1e+15:
                            tenpy.printf('Method converged due to gradient tolerance in upper direction',i,'iterations')
                            break
                    
                    prev_res = res

                end = time.time()
                
                res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                #print('Residual after convergence is',res)
                
                t_all+= end - start
                
                if converged:
                    break
            if tenpy.is_master_proc():
                # write to csv file
                if csv_writer is not None:
                    if method != 'NLS':
                        csv_writer.writerow([ R, k, i,t_all, res, converged])
                    else:
                        csv_writer.writerow([ R, k,iters,t_all, res, converged])

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_nls_arguments(parser)
    arg_defs.add_probability_arguments(parser)
    args, _ = parser.parse_known_args()
    
    # Set up CSV logging
    csv_path = join(results_dir, 'NewConvprob'+arg_defs.get_prob_file_prefix(args)+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    s = args.s
    order = args.order
    R = args.R
    r = args.r
    nls_tol = args.nls_tol
    grad_tol = args.grad_tol
    cg_tol = args.cg_tol
    num=args.num
    num_iter = args.num_iter
    tensor = args.tensor
    tlib = args.tlib
    Regu = args.regularization
    num_gen = args.num_gen
    num_init = args.num_init
    
    f_R = args.f_R
    l_R=args.l_R
    diag = args.diag
    num = args.num
    Arm = args.arm
    c = args.c
    tau = args.tau
    arm_iters=args.arm_iters
    
    
    if tlib == "numpy":
        import backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import backend.ctf_ext as tenpy

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow(['R','problem','iterations', 'time', 'residual','converged'])

    convprob(tenpy,tensor,s,f_R,l_R,num_iter,num_gen,args.probmethod,csv_writer,num_init,Regu,args)
