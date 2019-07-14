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

def convprob(tenpy,s,f_R,l_R,num_iter,num_gen,csv_writer=None,num_init = 10, method='NLS',num=1, Regu= 1e-05,cg_tol=1e-04,grad_tol = 1e-04,converged_tol = 5e-05):
    
    conv = []
    
    for R in range(f_R,l_R+1):
        
        converged_method=0.0

        for k in range(num_gen):
            a = tenpy.random((s,R))
            b = tenpy.random((s,R))
            c = tenpy.random((s,R))

            T = tenpy.einsum('ia,ja,ka->ijk', a,b,c)

            converged = 0

            for j in range(num_init):
                A = tenpy.random((s,R))
                P = A.copy()

                B = tenpy.random((s,R))
                Q = B.copy()

                C = tenpy.random((s,R))
                N = C.copy()

                X = [A,B,C]

                optimizer_list = {
                'DT': CP_DTALS_Optimizer(tenpy,T,X),
                'NLS': CP_fastNLS_Optimizer(tenpy,T,X,cg_tol,num)
                 }
                optimizer = optimizer_list[method]

                res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                print('Residual is',res)

                start = time.time()
                for i in range(num_iter):
                    delta = optimizer.step(Regu)

                    res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])

                    if res< converged_tol:
                        converged= 1
                        break

                end = time.time()
                
                res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                print('Residual after convergence is',res)

                if converged:
                    converged_method+=1
                    break



                """for i in range(als_iter):
                    Regu = 10**-6
                    tolerance = 10**-5

                    [P,Q,N]= stnd_ALS.dt_ALS_step(tenpy,T,P,Q,N,Regu)
                    res = ck.get_residual3(tenpy,T,P,Q,N)
                    if res<tolerance:
                        print('Iterations',i)
                        break

                end = time.time()


                print("Time taken for als",end-start)
                res = ck.get_residual3(tenpy,T,X[0],X[1],X[2])
                #print('state is',state)
                print('Residual with atol is',res)"""

        
        conv+=[converged_method/num_gen]
        if tenpy.is_master_proc():
                # write to csv file
                if csv_writer is not None:
                    csv_writer.writerow([ s, R, i, res, converged_method/num_gen ])
    
    print('conv=',conv)
    
    return conv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_nls_arguments(parser)
    arg_defs.add_probability_arguments(parser)
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
                'iterations', 'time', 'residual'
            ])

    convprob(tenpy,s,f_R,l_R,num_iter,num_gen,csv_writer,num_init,args.method,num,Regu,cg_tol,grad_tol,conv_tol)
