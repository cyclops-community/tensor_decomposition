from tensor_decomposition.CPD.NLS import fast_hessian_contract, CP_fastNLS_Optimizer
from tensor_decomposition.CPD.common_kernels import solve_sys, get_residual
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

def init_matrixmul(tenpy, m1,m2,m3, seed=1):
    I1 = tenpy.speye(m2)
    I2 = tenpy.speye(m1)
    I3 = tenpy.speye(m3)
    T = tenpy.einsum("lm,ik,nj->ijklmn", I1,I2,I3)
    T = T.reshape((m1*m3,m1*m2,m2*m3))
    O = T
    return [T, O]


def matmul(tenpy,m1,m2,m3,R,seed_start,tries,tol_init,tol_fin,method,csv_file):
    [T,O] = init_matrixmul(tenpy,m1,m2,m3,seed=1)
    conv_ALS = 0
    conv_GN = 0
    conv_Hyb = 0
    t = 0
    iters=0
    if method =='ALS':
        total_start = time.time()
        for j in range(tries):
            start = time.time()
            seed = 1301*seed_start +131*j
            np.random.seed(seed)
            s =[m1*m3,m1*m2,m2*m3]
            X= []
            for j in range(3):
                X.append(np.random.randn(s[j],R))
                
            num_iter = 250
            Regu = 1e-02

            opt = CP_DTALS_Optimizer(tenpy,T,X)
            
            
            for i in range(num_iter):

                X = opt.step(Regu)
                res = get_residual(tenpy,T,X)
                if res<tol_init:
                    tenpy.printf("Method Converged due to initialization tolerance in ",i,"iterations")
                    break
                    
            
            opt = CP_DTALS_Optimizer(tenpy,T,X)
            num_iter = 10000
            Regu = 1e-02
            
            
            for i in range(num_iter):
    
                X = opt.step(Regu)
    
                res = get_residual(tenpy,T,X)
                tenpy.printf("Residual is",res)
                if res < tol_fin:
                    tenpy.printf('Method converged due to tolerance defined for ALS',i,'iterations')
                    conv_ALS+=1
                    end = time.time()
                    iters += i
                    t+= end - start
                    break
                
                if Regu>1e-10:
                    Regu = Regu/2
        total_end = time.time()
        
        tenpy.printf("Number of trials converged",conv_ALS)
        if tenpy.is_master_proc():
                # write to csv file
                if csv_file is not None:
                    csv_writer.writerow([m1,m2,m3,R, method, tries,conv_ALS, iters/tries,total_end-total_start,t/tries,seed_start])
                    csv_file.flush()
        
    if method =='GN':
        cg_iters = 0
        total_start = time.time()
        for j in range(tries):
            start = time.time()
            seed = 1301*seed_start +131*j
            np.random.seed(seed)
            s =[m1*m3,m1*m2,m2*m3]
            X= []
            for j in range(3):
                X.append(np.random.randn(s[j],R))
                
            maxiter= 3*np.max(s)*R
            num_iter = 100
            Regu = 1e-02
            opt = CP_fastNLS_Optimizer(tenpy,T,X,maxiter,cg_tol=1e-03,num=0,diag=1,Arm=0,c=0.5,tau=0.5,arm_iters=10)


            for i in range(num_iter):
                
                [A,iters,flag] = opt.step(Regu)
                res = get_residual(tenpy,T,X)
                if res<tol_init:
                    tenpy.printf("Method Converged due to initialization tolerance in ",i,"iterations")
                    break
    
            maxiter= 3*np.max(s)*R
            num_iter = 300
            Regu = 1e-03
            varying=1
            lower = 1e-07
            upper = 1e-03
            fact= 2
            opt = CP_fastNLS_Optimizer(tenpy,T,X,maxiter,cg_tol=1e-03,num=0,diag=0,Arm=0,c=0.5,tau=0.5,arm_iters=10)
            decrease= True
            increase=False
                
            for i in range(num_iter):
    
                [A,iters,flag] = opt.step(Regu)
    
                res = get_residual(tenpy,T,X)
                tenpy.printf("Residual is",res)
                if res < tol_fin:
                    tenpy.printf('Method converged due to final tolerance',i,'iterations')
                    conv_GN+=1
                    end = time.time()
                    cg_iters += iters
                    t+= end - start
                    break
                
                if varying:
                    
                    if Regu <  lower:   
                        increase=True
                        decrease=False
    
                    if Regu > upper:
    
                        decrease= True
                        increase=False
    
    
                    if increase:
                        Regu = Regu*fact
    
                    elif decrease:
                        Regu = Regu/fact
        
        total_end = time.time()                
        tenpy.printf("Number of trials converged",conv_GN)
        if tenpy.is_master_proc():
            # write to csv file
            if csv_file is not None:
                csv_writer.writerow([m1,m2,m3,R, method, tries,conv_GN, cg_iters/tries,total_end-total_start,t/tries,seed_start])
                csv_file.flush()
            
    if method == 'HYB':
        total_start = time.time()
        cg_iters = 0
        for j in range(tries):
            start = time.time()
            seed = 1301*seed_start +131*j
            np.random.seed(seed)
            s =[m1*m3,m1*m2,m2*m3]
            X= []
            for j in range(3):
                X.append(np.random.randn(s[j],R))
                
            num_iter = 150
            Regu = 1e-02

            opt = CP_DTALS_Optimizer(tenpy,T,X)
            
            
            for i in range(num_iter):

                X = opt.step(Regu)
                res = get_residual(tenpy,T,X)
                if res<tol_init:
                    tenpy.printf("Method Converged due to initialization tolerance in ",i,"iterations")
                    break
                    
            maxiter= 3*np.max(s)*R
            num_iter = 300
            Regu = 1e-02
            varying=1
            lower = 1e-07
            upper = 1e-03
            fact= 2
            opt = CP_fastNLS_Optimizer(tenpy,T,X,maxiter,cg_tol=0.5,num=0,diag=0,Arm=1,c=0.5,tau=0.5,arm_iters=10)
            decrease= True
            increase=False
                
            for i in range(num_iter):
    
                [A,iters,flag] = opt.step(Regu)
    
                res = get_residual(tenpy,T,X)
                tenpy.printf("Residual is",res)
                if res < tol_fin:
                    tenpy.printf('Method converged due to final tolerance',i,'iterations')
                    conv_Hyb+=1
                    end = time.time()
                    cg_iters+= iters
                    t+= end - start
                    break
                
                if varying:
                    
                    if Regu <  lower:   
                        increase=True
                        decrease=False
    
                    if Regu > upper:
    
                        decrease= True
                        increase=False
    
    
                    if increase:
                        Regu = Regu*fact
    
                    elif decrease:
                        Regu = Regu/fact
                        
        total_end = time.time()
        tenpy.printf("Number of trials converged",conv_Hyb)
        if tenpy.is_master_proc():
            # write to csv file
            if csv_file is not None:
                csv_writer.writerow([m1,m2,m3,R, method, tries,conv_Hyb, cg_iters/tries,total_end-total_start,t/tries,seed_start])
                csv_file.flush()

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
        help='choose tensor library to test, choose between numpy and ctf (default: numpy)')
    parser.add_argument(
    '--R',
    type=int,
    default=4,
    metavar="int",
    help="Rank for matrix multiplication tensor,default is 4")    
    parser.add_argument(
    '--m1',
    type=int,
    default=3,
    metavar="int",
    help="first dimension for matmul, default is 3")
    parser.add_argument(
    '--m2',
    type=int,
    default=3,
    metavar="int",
    help="second dimension for matmul, default is 3")
    parser.add_argument(
    '--m3',
    type=int,
    default=3,
    metavar="int",
    help="third dimension for matmul, default is 3")
    parser.add_argument(
    '--tol-init',
    type=float,
    default=0.01,
    metavar="float",
    help="tolerance for initialization convergence, default is 0.01")
    parser.add_argument(
    '--tol-fin',
    type=float,
    default=1e-08,
    metavar="float",
    help="tolerance for final convergence, default is 1e-08")
    parser.add_argument(
        '--method',
        default='HYB',
        metavar='string',
        choices=[
            'GN',
            'ALS',
            'HYB'
            ],
        help='choose the optimization method: GN,AL,HYB (default: HYB)')
    parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar="int",
    help="seed to generate random initializations")
    parser.add_argument(
    '--tries',
    type=int,
    default=5,
    metavar="int",
    help="number of trials")
    
    
    
    args, _ = parser.parse_known_args()
    
    # Set up CSV logging
    csv_path = join(results_dir, 'Matmul'+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    
    tlib = args.tlib
    R = args.R
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    tries = args.tries
    tol_init = args.tol_init
    tol_fin=args.tol_fin
    method = args.method
    seed = args.seed
    
    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        
    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'Dim1','Dim2','Dim3','Rank','Method', 'Trials', 'Converged', 'Av iterations (after initial step)','Total Time Taken','Av Time for converged iter','seed'
            ])

    matmul(tenpy,m1,m2,m3,R,seed,tries,tol_init,tol_fin,method,csv_file)
