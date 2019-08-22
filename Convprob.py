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

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def convprob(tenpy,tensor,s,f_R,l_R,num_iter,num_gen,csv_writer=None,num_init = 10, method='DT',num=1, Regu= 1e-05,cg_tol=1e-04,grad_tol = 1e-04,converged_tol = 5e-05):
    
    conv = []
    orig_Regu = Regu
    
    for R in range(f_R,l_R+1):

        for k in range(num_gen):
            if tensor == 'negrandom':
                a = tenpy.random((s,R)) - tenpy.random((s,R))
                b = tenpy.random((s,R)) - tenpy.random((s,R))
                c = tenpy.random((s,R)) - tenpy.random((s,R))
                
            a = tenpy.random((s,R)) 
            b = tenpy.random((s,R))
            c = tenpy.random((s,R))

            T = tenpy.einsum('ia,ja,ka->ijk', a,b,c)

            for j in range(num_init):
                total_iters = 0
                converged = 0
                t_all = 0.0
                
                A = tenpy.random((s,R))
                P = A.copy()

                B = tenpy.random((s,R))
                Q = B.copy()

                C = tenpy.random((s,R))
                N = C.copy()

                X = [A,B,C]
                
                Regu = orig_Regu

                optimizer_list = {
                'DT': CP_DTALS_Optimizer(tenpy,T,X),
                'NLS': CP_fastNLS_Optimizer(tenpy,T,X,cg_tol,num)
                 }
                optimizer = optimizer_list[method]

                prev_res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                #print('Residual is',prev_res)
                
                start = time.time()
                for i in range(num_iter):
                    if method == 'NLS':
                        [delta,iters] = optimizer.step2(Regu)
                        total_iters+= iters
                        
                    else:
                        delta = optimizer.step(Regu)
                 
                    
                    res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                    
                    if abs(prev_res - res)< 1e-07:
                        break
                    
                    if method == "NLS" :
                        if fitness > 0.999:
                            flag = True
            
                        if flag:   
                            Regu = 1e-05
                
                        else:
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
                            
                    if res< converged_tol:
                        converged= 1
                        break
                    
                    prev_res = res

                end = time.time()
                
                res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                #print('Residual after convergence is',res)
                
                t_all+= end - start
                    
                if tenpy.is_master_proc():
                    # write to csv file
                    if csv_writer is not None:
                        if method != 'NLS':
                            csv_writer.writerow([ R, k, j, i,t_all, res, converged])
                        else:
                            csv_writer.writerow([ R, k, j, total_iters,t_all, res, converged])

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_nls_arguments(parser)
    arg_defs.add_probability_arguments(parser)
    args, _ = parser.parse_known_args()
    
    # Set up CSV logging
    csv_path = join(results_dir, 'Convprob'+arg_defs.get_prob_file_prefix(args)+'.csv')
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
    conv_tol = args.conv_tol
    f_R = args.f_R
    l_R=args.l_R
    
    
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
            csv_writer.writerow([
                'R','problem', 'initialization','iterations', 'time', 'residual','converged'
            ])

    convprob(tenpy,tensor,s,f_R,l_R,num_iter,num_gen,csv_writer,num_init,args.method,num,Regu,cg_tol,grad_tol,conv_tol)
