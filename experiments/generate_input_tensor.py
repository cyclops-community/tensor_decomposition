import numpy as np

import tensor_decomposition
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import argparse
import tensor_decomposition.utils.arg_defs as arg_defs

def generate_tensor(tenpy, args):  

    T = None  
    O = None 
    
    if args.load_tensor != '':
        T = tenpy.load_tensor_from_file(args.load_tensor+'tensor.npy')
        O = None
    elif args.tensor == "random":      
        tenpy.printf("Testing random tensor")
        [T,O] = synthetic_tensors.rand(tenpy,order,args.s,R,args.sp_frac,args.seed)
    #################################################
    elif args.tensor == "noisy_tensor":
        dims = [args.s] * args.order 
        if args.type_noisy_tensor == "new_model":
            T_true, T, cov_empirical,cov_pinv_empirical, M_empirical_pinv =                   synthetic_tensors.generate_tensor_with_noise_new_model(tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed) 
            O = None
        else: 
            T_true, T, cov_empirical,cov_pinv_empirical,_,_,M_empirical_pinv,_=                   synthetic_tensors.generate_tensor_with_noise_old_model(tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed) 
            O = None
    #################################################################
    elif args.tensor == "MGH":
        T = tenpy.load_tensor_from_file("MGH-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif args.tensor == "SLEEP":
        T = tenpy.load_tensor_from_file("SLEEP-16.npy")
        T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
        O = None
    elif args.tensor == "random_col":
        dims = [args.s] * args.order 
        T_true, O, T, cov_empirical,cov_pinv_empirical, M_empirical_pinv  = synthetic_tensors.collinearity_tensor(tenpy, args.s, args.order, args.R, args.k, args.epsilon, args.col, args.seed)
        O = None
    elif args.tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)
        O = None
    elif args.tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
        O = None

    tenpy.printf("The shape of the input tensor is: ", T.shape)
    
    return T_true,T, O, cov_empirical,cov_pinv_empirical, M_empirical_pinv

