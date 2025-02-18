import numpy as np
import argparse
import tensor_decomposition.utils.arg_defs as arg_defs


def generate_initial_guess(tenpy, T,args):  


    A_ini = []
    
    tenpy.seed(args.seed)
    if args.decomposition == "CP":
        for i in range(T.ndim):
            A_ini.append(tenpy.random((T.shape[i], args.R_app)))
    else:
        for i in range(T.ndim):
            A_ini.append(tenpy.random((T.shape[i], args,args.hosvd_core_dim[i])))
    return A_ini

