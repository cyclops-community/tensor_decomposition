import numpy as np
import sys
import time
import os
import csv
from pathlib import Path
from os.path import dirname, join
import tensor_decomposition
#import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import argparse
import tensor_decomposition.utils.arg_defs as arg_defs
import csv
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer
##########################################
#import Generate_plots_old
#import error_computation
from scipy.linalg import svd
import numpy.linalg as la
import matplotlib.pyplot as plt
import copy
import random
from generate_initial_guess import generate_initial_guess
##########################################
parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


def CP_ALS(tenpy,
           T_true,
           A_ini,
           T,
           O,
           args,
           cov_empirical,
           cov_pinv_empirical,
           M_empirical_pinv,
           csv_file=None,
           Regu=None,
           method='DT',
           res_calc_freq=1,
           tol=1e-05):

##

    final_residuals = []
    all_residuals = []
    all_norm_mahalanobis_empirical = []
    for run in range(args.num_runs):
        #A = copy.deepcopy(A_ini)  
        A = [np.random.rand(T.shape[i], args.R) for i in range(T.ndim)]
        #A = generate_initial_guess(tenpy, T, args)
        print(f"Run {run + 1}/{args.num_runs}")
        
        residuals = []
        norm_mahalanobis_emp = []
    ##
        flag_dt = True
    
        if csv_file is not None:
            csv_writer = csv.writer(csv_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            
        if Regu is None:
            Regu = 0
    
        normT = tenpy.vecnorm(T)
    
        time_all = 0.
        if args is None:
            optimizer = CP_DTALS_Optimizer(tenpy, T, A,args)
        else:
            optimizer_list = {
                'DT': CP_DTALS_Optimizer(tenpy, T, A,args),
                'PP': CP_PPALS_Optimizer(tenpy, T, A, args),
            }
            optimizer = optimizer_list[method]
    
        fitness_old = 0
        for i in range(args.num_iter):
    
            if i % res_calc_freq == 0 or i == args.num_iter - 1 or not flag_dt:
                if args.fast_residual and i != 0:
                   res = optimizer.compute_fast_residual()
                else:
                    if args.sp and O is not None:
                        res = get_residual_sp(tenpy,O,T,A)
                    else:
                        res = get_residual(tenpy, T, A)
                        ##
                residuals.append(res)
                fitness = 1 - res / normT
       #########################################################################
                T_reconstructed = tenpy.zeros(T.shape)
                T_reconstructed = T_reconstructed + tenpy.einsum('ir,jr,kr->ijk', *A)
                diff = T_true - T_reconstructed 
                norm_mahalanobis_empirical = np.einsum('ip,jq,kr,ijk,pqr->',  *M_empirical_pinv,  diff, diff)
                norm_mahalanobis_emp.append(norm_mahalanobis_empirical)
                ####
        ################################################################################################
                if args.calc_cond and R < 15 and tenpy.name() == 'numpy':
                    cond = compute_condition_number(tenpy, A)
                    if tenpy.is_master_proc():
                        print("[", i, "] Residual is", res, "fitness is: ", fitness)
                        #print(f"[{i}] Residual: {res}, Fitness: {fitness}, Mahalanobis Norm (True):                                           {norm_mahalanobis_true}, Mahalanobis Norm (Empirical): {norm_mahalanobis_empirical}")
                        # write to csv file
                        if csv_file is not None:
                            ###############################################
                            csv_writer.writerow([i, time_all, res, fitness, flag_dt,cond,                                    norm_mahalanobis_empirical, err_UW])
                            csv_file.flush()
                            
                else:
                    if tenpy.is_master_proc():
                        print("[", i, "] Residual is", res, "fitness is: ", fitness)
                        # write to csv file
                        if csv_file is not None:
                            csv_writer.writerow([i, time_all, res, fitness, flag_dt,                                        norm_mahalanobis_empirical, err_UW])
                            ####################################################
                            csv_file.flush()
    
            if res < tol:
                print('Method converged in', i, 'iterations')
                break
            t0 = time.time()
            if method == 'PP':
                A, pp_restart = optimizer.step(Regu)
                flag_dt = not pp_restart
            else:
                A = optimizer.step(Regu)
            t1 = time.time()
            tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")
    
            time_all += t1 - t0
            fitness_old = fitness
    ##
        final_residuals.append(residuals[-1])
        all_residuals.append(residuals)
        all_norm_mahalanobis_empirical.append(norm_mahalanobis_emp)
        
        tenpy.printf(method + " method took", time_all, "seconds overall")
    
        if args.save_tensor:
            folderpath = join(results_dir, arg_defs.get_file_prefix(args))
            save_decomposition_results(T, A, tenpy, folderpath)
    ##
    best_run_index = np.argmin(final_residuals)
    best_run_residual = all_residuals[best_run_index]
    best_run_norm_mahalanobis_empirical = all_norm_mahalanobis_empirical[best_run_index]
    iterations = np.arange(1, len(best_run_residual) + 1)
    final_residuals = np.sort(final_residuals)[::-1]
    
    min_length = min(len(residuals) for residuals in all_residuals)
    # Truncate each run's residuals to the minimum length
    truncated_residuals = [residuals[-min_length:] for residuals in all_residuals]
    truncated_norm_mahalanobis_empirical = [rr[-min_length:] for rr in all_norm_mahalanobis_empirical]
    #truncated_norm_mahalanobis_true = [rr[-min_length:] for rr in all_norm_mahalanobis_true]
    # Convert to a NumPy array
    truncated_residuals = np.array(truncated_residuals)  
    truncated_norm_mahalanobis_empirical = np.array(truncated_norm_mahalanobis_empirical)
    #truncated_norm_mahalanobis_true = np.array(truncated_norm_mahalanobis_true)
    mean_residuals = np.mean(truncated_residuals, axis=0) # Compute mean and standard deviation across runs for each                                                                          iteration
    mean_norm_mahalanobis_empirical = np.mean(truncated_norm_mahalanobis_empirical, axis=0)
    std_residuals = np.std(truncated_residuals, axis=0) 

     
    return best_run_residual, best_run_norm_mahalanobis_empirical, final_residuals, mean_residuals, std_residuals
