import numpy as np
import sys
import time
import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join
import tensors.synthetic_tensors as synthetic_tensors
import tensors.real_tensors as real_tensors
import argparse
import arg_defs as arg_defs
import csv

from utils import save_decomposition_results

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def CP_NLS(tenpy,A,T,O,num_iter,csv_file=None,Regu=None,method='NLS',args=None,res_calc_freq=1):

    from CPD.common_kernels import get_residual
    from CPD.NLS import CP_fastNLS_Optimizer


    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    if Regu is None:
        Regu = 0
        
    if args.varying:
        decrease= True
        increase=False
    
    iters = 0
    count = 0

    normT = tenpy.vecnorm(T)
    
    if args.maxiter == 0:
        args.maxiter = sum(T.shape)*R
    
    time_all = 0.
    if method == 'DT':
        method = 'NLS'
        optimizer = CP_fastNLS_Optimizer(tenpy,T,A,args)
    else:
        optimizer_list = {
            'NLS': CP_fastNLS_Optimizer(tenpy,T,A,args)
        }
        optimizer = optimizer_list[method]

    fitness_old = 0
    prev_res = np.finfo(np.float32).max
    for i in range(num_iter):

        if i % res_calc_freq == 0 or i==num_iter-1 :
            if args.sp:
                res = get_residual_sp(tenpy,O,T,A)
            else:
                res = get_residual(tenpy,T,A)
            fitness = 1-res/normT

            if tenpy.is_master_proc():
                print("[",i,"] Residual is", res, "fitness is: ", fitness)
                # write to csv file
                if csv_file is not None:
                    if method == 'NLS':
                        csv_writer.writerow([iters, time_all, res, fitness])
                    else:
                        csv_writer.writerow([i, time_all, res, fitness])
                    csv_file.flush()
        
           
        if res<args.nls_tol:
            tenpy.printf('Method converged due to residual tolerance in',i,'iterations')
            
            break
        t0 = time.time()
        
        
        if method == 'NLS':
            [A,iters] = optimizer.step(Regu)
        else:
            A = optimizer.step(Regu)
        count+=1
            
        t1 = time.time()
        tenpy.printf("[",i,"] Sweep took", t1-t0,"seconds")
        
        time_all += t1-t0
        
        if method == 'NLS':
            if optimizer.g_norm < args.grad_tol:
                tenpy.printf('Method converged due to gradient tolerance in',i,'iterations')
                break
        
        #fitness_old = fitness
        
        
        if args.varying:
            if Regu <  args.lower:   
                increase=True
                decrease=False
                
            if Regu > args.upper:

                decrease= True
                increase=False
                    
                    
            if increase:
                Regu = Regu*args.varying_fact
                    
            elif decrease:
                Regu = Regu/args.varying_fact
                
        
    
    tenpy.printf(method+" method took",time_all,"seconds overall")
    
    

    if args.save_tensor:
        folderpath = join(results_dir, arg_defs.get_file_prefix(args))
        save_decomposition_results(T,A,tenpy,folderpath)

    return A



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_lrdt_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_nls_arguments(parser)
    arg_defs.add_col_arguments(parser)
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
    nls_iter = args.nls_iter
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor = args.tensor
    tlib = args.tlib
    

    if tlib == "numpy":
        import backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin();

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual', 'fitness'
            ])

    tenpy.seed(args.seed)

    if args.load_tensor is not '':
        T = tenpy.load_tensor_from_file(args.load_tensor+'tensor.npy')
        O = None
    elif tensor == "random":
        if args.decomposition == "CP":
            tenpy.printf("Testing random tensor")
            [T,O] = synthetic_tensors.init_rand(tenpy,order,s,R,sp_frac,args.seed)
        if args.decomposition == "Tucker":
            tenpy.printf("Testing random tensor")
            shape = s * np.ones(order).astype(int)
            T = tenpy.random(shape)
            O = None
    elif tensor == "random_col":
        [T,O] = synthetic_tensors.init_collinearity_tensor(tenpy, s, order, R, args.col, args.seed)
    elif tensor == "mom_cons":
        tenpy.printf("Testing order 4 momentum conservation tensor")
        T = synthetic_tensors.init_mom_cons(tenpy,s)
        O = None
        sp_res = False
    elif tensor == "mom_cons_sv":
        tenpy.printf("Testing order 3 singular vectors of unfolding of momentum conservation tensor")
        T = synthetic_tensors.init_mom_cons_sv(tenpy,s)
        O = None
        sp_res = False
    elif tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
        O = None
    elif tensor == "coil100":
        T = real_tensors.coil_100(tenpy)
        O = None
    elif tensor == "timelapse":
        T = real_tensors.time_lapse_images(tenpy)
        O = None
    elif tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)
        O = None
    elif tensor == "embedding":
        T = real_tensors.get_bert_embedding_tensor(tenpy)
        O = None
    elif tensor == "bert-param":
        T = real_tensors.get_bert_weights_tensor(tenpy)
        O = None
    elif tensor == "mm":
        tenpy.printf("Testing matrix multiplication tensor")
        [T,O] = synthetic_tensors.init_mm(tenpy,s,R,args.seed)
    elif tensor == "negrandom":
        tenpy.printf("Testing random tensor with negative entries")
        [T,O] = synthetic_tensors.init_neg_rand(tenpy,order,s,R,sp_frac,args.seed)
        
    elif tensor == "randn":
        tenpy.printf("Testing random tensor with normally distributed entries")
        [T,O] = synthetic_tensors.init_randn(tenpy,order,s,R,sp_frac,args.seed)
        
    tenpy.printf("The shape of the input tensor is: ", T.shape)

    Regu = args.regularization

    A = []
    
    if args.load_tensor is not '':
        for i in range(T.ndim):
            A.append(tenpy.load_tensor_from_file(args.load_tensor+'mat'+str(i)+'.npy'))
    elif args.hosvd != 0:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((args.hosvd_core_dim[i],R)))
        elif args.decomposition == "Tucker":
            from Tucker.common_kernels import hosvd
            A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
    else:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], R)))
        else:
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], args,hosvd_core_dim[i])))

    
    CP_NLS(tenpy,A,T,O,num_iter,csv_file,Regu,args.method ,args, args.res_calc_freq)
    
