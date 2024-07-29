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
import numpy.linalg as la

from CPD.common_kernels import get_residual,get_residual_sp,equilibrate,solve_sys,normalise

from utils import save_decomposition_results

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def compute_pseudo(tenpy,A,thresh,k):
	A_pseudo = []
	Vts = []
	sigs = []
	for i in range(len(A)):
		if i ==k :
			A_pseudo.append(tenpy.zeros(A[k].shape))
			sigs.append(tenpy.zeros(A[k].shape[1]))
			Vts.append(tenpy.zeros((A[k].shape[1],A[k].shape[1])))
		else:
			if thresh> min(A[i].shape[0],A[i].shape[1]):
				thresh = min(A[i].shape[0],A[i].shape[1])
			U,s,Vt = tenpy.svd(A[i])
			s_new = s.copy()
			#s_new = tenpy.zeros(s.shape)
			p = 0
			for j in range(s.shape[0]) :
				if j<thresh :
				#if s[0]/s[j] < thresh:
					s_new[j] = 1/s[j]
			A_pseudo.append(tenpy.einsum('is,s,sj->ij',U,s_new,Vt))
			sigs.append(s_new*s)
			Vts.append(Vt)

	return A_pseudo,Vts,sigs

def init_matrixmul(tenpy, m1,m2,m3, seed=1):
    I1 = tenpy.speye(m2)
    I2 = tenpy.speye(m1)
    I3 = tenpy.speye(m3)
    T = tenpy.einsum("lm,ik,nj->ijklmn", I1,I2,I3)
    T = T.reshape((m1*m3,m1*m2,m2*m3))
    O = T
    return [T, O]

def bad_cond_matrix(tenpy,s,R,k=1e-07,seed=123):
    A = []
    for i in range(2):
        A.append(tenpy.random((s,R)))
    for i in range(2):
        U,sig,Vt = tenpy.svd(A[i])
        sig[-1] = sig[0]*k
        A[i] = U@np.diag(sig)@Vt
    T = tenpy.einsum('ir,jr->ij',A[0],A[1])
    return T

#Get a vector and generate the complement space
def construct_complement(a):
    B= np.random.rand(a.shape[0],a.shape[0])
    B[:,0]=a
    Q,R = la.qr(B)
    a_Complement = Q[:,1:]
    return a_Complement
    
def construct_Terracini(A):
    R = A[0].shape[1]
    s = A[0].shape[0]
    order = len(A)
    Us = []
    for r in range(R):
        U_i = np.zeros((s**order,order*(s-1) +1))
        if order == 3 :
	        U_i[:,0] = np.einsum('i,j,k->ijk',A[0][:,r],A[1][:,r],A[2][:,r]).reshape(-1)
	        U_i[:,1:s] = np.kron(np.kron(construct_complement(A[0][:,r]),A[1][:,r].reshape(-1,1)), A[2][:,r].reshape(-1,1))
	        U_i[:,s:2*s-1]=np.kron(np.kron(A[0][:,r].reshape(-1,1),construct_complement(A[1][:,r])), A[2][:,r].reshape(-1,1))
	        U_i[:,2*s-1:3*s-2]=np.kron(np.kron(A[0][:,r].reshape(-1,1),A[1][:,r].reshape(-1,1)),construct_complement(A[2][:,r]))
        elif order == 4:
        	U_i[:,0] = np.einsum('i,j,k,l->ijkl',A[0][:,r],A[1][:,r],A[2][:,r], A[3][:,r]).reshape(-1)
	        U_i[:,1:s] = np.kron(np.kron(np.kron(construct_complement(A[0][:,r]),A[1][:,r].reshape(-1,1)), A[2][:,r].reshape(-1,1)),A[3][:,r].reshape(-1,1) )
	        U_i[:,s:2*s-1] = np.kron(np.kron(np.kron(A[0][:,r].reshape(-1,1),construct_complement(A[1][:,r])), A[2][:,r].reshape(-1,1)) ,A[3][:,r].reshape(-1,1))
	        U_i[:,2*s-1:3*s-2]= np.kron(np.kron(np.kron(A[0][:,r].reshape(-1,1),A[1][:,r].reshape(-1,1)),construct_complement(A[2][:,r])) ,A[3][:,r].reshape(-1,1))
	        U_i[:,3*s-2:4*s-3] = np.kron(np.kron(np.kron(A[0][:,r].reshape(-1,1),A[1][:,r].reshape(-1,1)),A[2][:,r].reshape(-1,1)) ,construct_complement(A[3][:,r]))
        else:
        	print("CONDITION NUMBER CALCULATION NOT GENERALIZED")
        Us.append(U_i)
    U = np.zeros((s**order,R*(order*(s-1) +1)))
    for r in range(R):
        U[:,r*(order*(s-1)+1):(r+1)*(order*(s-1)+1)] = Us[r]
    return U

def Compute_condition_number(A):
    A_core =[]
    for i in range(len(A)):
    	Q,R_ = la.qr(A[i])
    	A_core.append(R_)
    normalised_CP_f = normalise(tenpy,A_core)
    #equilibrated_CP_f = equilibrate(tenpy,A_core)
    U = construct_Terracini(normalised_CP_f)
    #U=construct_Terracini(equilibrated_CP_f)
    U_,sig,Vt= la.svd(U)
    print('CPD condition number is',1/sig[-1])
    return 1/sig[-1]


def CP_Mahalanobis(tenpy,A,T,O,num_iter,thresh,csv_file=None,Regu=None,args=None,res_calc_freq=1):
	from CPD.standard_ALS import CP_DTALS_Optimizer

	if csv_file is not None:
		csv_writer = csv.writer(
			csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	if Regu is None:
		Regu = 0
	
	iters = 0
	count = 0
	
	
	time_all = 0.
	normT = tenpy.vecnorm(T)
	method = "M-norm"
	fitness_old = 1
	prev_res = np.finfo(np.float32).max

	if args.calc_cond:
		c= Compute_condition_number(A)
	else:
		c=0

	#opt = CP_DTALS_Optimizer(tenpy, T, A,args)

	#for i in range(2):
	#	A = opt.step(Regu)

	for k in range(1,num_iter):
		if k % res_calc_freq == 0 or k==num_iter-1 :
			if args.sp:
				res = get_residual_sp(tenpy,O,T,A)
			else:
				res = get_residual(tenpy,T,A)

			fitness = 1-res/normT

			if tenpy.is_master_proc():
				print("[",k,"] Residual is", res, "fitness is: ", fitness)
				# write to csv file
				if csv_file is not None:
					csv_writer.writerow([method,k, time_all, res, fitness,c])
					csv_file.flush()
		t0 = time.time()
		
		for i in range(len(A)):
			#A = equilibrate(tenpy,A)
			A = normalise(tenpy,A)
			A_pseudo,Vts,sigs_inv = compute_pseudo(tenpy,A,thresh,i)
			M = tenpy.ones((A[0].shape[1],A[0].shape[1]))
			if args.sp:
				lst = A_pseudo[:]
				lst[i] = tenpy.zeros(A[i].shape)
				tenpy.MTTKRP(T,lst,i)
			else:
				T_inds = "".join([chr(ord('a')+m) for m in range(T.ndim)])
				einstr = ""
				A2 = []
				for j in range(len(A)):
				    if j != i:
				        einstr += chr(ord('a')+j) + chr(ord('a')+T.ndim) + ','
				        A2.append(A_pseudo[j])
				einstr += T_inds + "->" + chr(ord('a')+i) + chr(ord('a')+T.ndim)
				A2.append(T)
				rhs = tenpy.einsum(einstr,*A2)
			for j in range(len(A)):
				if j != i:
					M *= tenpy.einsum('sr,s,sz->rz',Vts[j],sigs_inv[j],Vts[j])
			if args.sp:
				A[i] = solve_sys(tenpy, M, lst[i])
			else:
				A[i] = solve_sys(tenpy, M, rhs)
		#print('thresh is',thresh)
		if args.reduce_thresh:
			if k>0 and k%args.reduce_thresh_freq==0 and thresh>=0:
				thresh = thresh - 1

		if args.calc_cond:
			c= Compute_condition_number(A)
		else:
			c= 0

		t1 = time.time()

		tenpy.printf("[",k,"] Sweep took", t1-t0,"seconds")

		time_all += t1-t0

		if res<args.tol:
			tenpy.printf('Method converged due to residual tolerance in',k,'iterations')
			break

		if fitness>args.fit:
			tenpy.printf('Method converged due to fitness tolerance in',k,'iterations')
			break

	#opt = CP_DTALS_Optimizer(tenpy, T, A,args)

	#for i in range(5):
	#	A = opt.step(Regu)
	#	if A[0].shape[1]<=15:
	#		c= Compute_condition_number(A)
	#	else:
	#		c=0
	#	res = get_residual(tenpy,T,A)
	#	fitness = 1-res/normT
	#	if tenpy.is_master_proc():
	#			print("[",i,"] Residual is", res, "fitness is: ", fitness)
	tenpy.printf(method+" method took",time_all,"seconds overall")
	

	if args.save_tensor:
		folderpath = join(results_dir, arg_defs.get_file_prefix(args))
		save_decomposition_results(T,A,tenpy,folderpath)

	return A


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	arg_defs.add_general_arguments(parser)
	arg_defs.add_sparse_arguments(parser)
	arg_defs.add_col_arguments(parser)
	args, _ = parser.parse_known_args()

	# Set up CSV logging
	csv_path = join(results_dir, 'Mahalanobis-'+args.tensor+'-order-'+str(args.order)+'-s-'+str(args.s)+'-R-'
		+str(args.R)+'-R_app-'+str(args.R_app)+'-thresh-'+str(args.thresh)+'.csv')
	is_new_log = not Path(csv_path).exists()
	csv_file = open(csv_path, 'a')#, newline='')
	csv_writer = csv.writer(
		csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	s = args.s
	order = args.order
	R = args.R
	R_app = args.R_app
	num_iter = args.num_iter
	sp_frac = args.sp_fraction
	tensor = args.tensor
	tlib = args.tlib
	thresh = args.thresh
	

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
				'method','iterations', 'time', 'residual', 'fitness','cond_num'
			])

	tenpy.seed(args.seed)
	

	if args.load_tensor is not '':
		T = tenpy.load_tensor_from_file(args.load_tensor+'tensor.npy')
		O = None
	elif tensor == "random":
		tenpy.printf("Testing random tensor")
		[T,O] = synthetic_tensors.init_rand(tenpy,order,s,R,sp_frac,args.seed)

	elif tensor == "MGH":
		T = tenpy.load_tensor_from_file("MGH-16.npy")
		T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
		O = None
	elif tensor == "SLEEP":
		T = tenpy.load_tensor_from_file("SLEEP-16.npy")
		T = T.reshape(T.shape[0]*T.shape[1], T.shape[2],T.shape[3],T.shape[4])
		O = None
	elif tensor == "random_col":
		[T,O] = synthetic_tensors.init_collinearity_tensor(tenpy, s, order, R, args.col, args.seed)
	elif tensor =="bad_cond":
		T = bad_cond_matrix(tenpy,s,R,args.seed)
		O = None
	elif tensor == "scf":
		T = real_tensors.get_scf_tensor(tenpy)
		O = None
	elif tensor == "amino":
		T = real_tensors.amino_acids(tenpy)
		O = None

	tenpy.printf("The shape of the input tensor is: ", T.shape)

	Regu = args.regularization

	A = []
	
	tenpy.seed(args.seed)
	if args.load_tensor is not '':
		for i in range(T.ndim):
			A.append(tenpy.load_tensor_from_file(args.load_tensor+'mat'+str(i)+'.npy'))
	elif args.hosvd != 0:
		if args.decomposition == "CP":
			for i in range(T.ndim):
				A.append(tenpy.random((args.hosvd_core_dim[i],R_app)))
		elif args.decomposition == "Tucker":
			from Tucker.common_kernels import hosvd
			A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
	else:
		if args.decomposition == "CP":
			for i in range(T.ndim):
				A.append(tenpy.random((T.shape[i], R_app)))
		else:
			for i in range(T.ndim):
				A.append(tenpy.random((T.shape[i], args,hosvd_core_dim[i])))
	CP_Mahalanobis(tenpy,A,T,O,num_iter,thresh, csv_file,Regu,args, args.res_calc_freq)

    
	
