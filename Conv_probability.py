# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:29:57 2019

@author: navjo
"""
import numpy as np
import sys
import time
import CPD.common_kernels as ck
import CPD.lowr_ALS3 as lowr_ALS
import CPD.standard_ALS3 as stnd_ALS
import argparse
import arg_defs as arg_defs
import csv
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()


    s = args.s
    R = args.R
    upper_rank = args.upper_rank
    max_iter = args.max_iter
    upper_rank = args.upper_rank
    lower_rank = args.lower_rank
    num_init = args.num_init
    num_gen = args.num_gen
    res_criteria = args.res_criteria
    
    Regu = args.regularization * tenpy.eye(R,R)
    
    tlib = args.tlib

    if tlib == "numpy":
        import backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import backend.ctf_ext as tenpy
    else:
        print("ERROR: Invalid --tlib input")

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args) :
            print( arg+':', getattr(args, arg))
            
            
    probab_conv(s,lower_rank,upper_rank,num_init , num_gen, max_iter , res_criteria,Regu)
            
def probab_conv(s,lower_rank,upper_rank,num_init =10, num_gen=20, max_iter = 10000, res_criteria=10**-6,Regu):
    probab = []

    for R in range(lower_rank,upper_rank+1):
        converged = 0
        
        for k in range(num_gen):
            flag = 0
            
            #Setting up Random Tensor
            a = tenpy.random(s,R)
    
            b = tenpy.random(s,R)
    
            c = tenpy.random(s,R)    
    
            T = tenpy.einsum('ia,ja,ka->ijk', a,b,c)
    
    
            for j in range(num_init):
    
                #Random Initialisation
    
                A = tenpy.random(s,R)
                
                B = tenpy.random(s,R)
     
                C = tenpy.random(s,R)
    
                ##Equilibrate
                for i in range(max_iter):
    
                    [A,B,C]= stnd_ALS.dt_ALS_step(tenpy,T,A,B,C,Regu)
    
                res = ck.get_residual3(tenpy,T,A,B,C)
                
                if res <res_criteria:
                    flag =1
                    converged+=1
                if flag == 1:
                    break
        probab+= [converged/(num_gen)]
        
        print('probability of a Tensor converging for rank',R,'is =',converged/(num_gen))

        
        
    return probab


