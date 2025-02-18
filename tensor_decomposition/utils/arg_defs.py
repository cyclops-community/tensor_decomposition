def add_general_arguments(parser):

    parser.add_argument(
        '--experiment-prefix',
        '-ep',
        type=str,
        default='',
        required=False,
        metavar='str',
        help='Output csv file name prefix (default: None)')
    ############################################################
    parser.add_argument(
        '--top_sin_val',
        type=int,
        default=1,
        metavar='int',
        help='Input the number of top singular values of empirical covariance (default: 3)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-4,
        metavar='float',
        help='Noise level (default: 1e-4)')
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        metavar='int',
        help='Input rank of noise tensor (default: 3)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.5,
        metavar='float',
        help='Input the rate of singular value decay (default: 1)')
    parser.add_argument(
        '--num_runs',
        type=int,
        default=5,
        metavar='int',
        help='Input number of runs (default: 5)')
    parser.add_argument(
        '--type_noisy_tensor',
        default="old_model",
        metavar='string',
        choices=[
            'old_model',
            'new_model'
            ],
        help='choose type of noisy tensor to test, available: old_model, new_model (default: old_model)')
    ############################################################
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        metavar='int',
        help='Tensor order (default: 3)')
    parser.add_argument(
        '--s',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in each dimension (default: 64)')
    parser.add_argument(
        '--R',
        type=int,
        default=10,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--r',
        type=int,
        default=10,
        metavar='int',
        help='Update rank size (default: 10)')
    parser.add_argument(
        '--R-app',
        type=int,
        default=10,
        metavar='int',
        help='Approximate rank (default: None here but changed to R)')
    parser.add_argument(
        '--num-iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations (default: 10)')
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.0000001,
        metavar='float',
        help='regularization (default: 0.0000001)')
    parser.add_argument(
        '--tensor',
        #default="random",
        #default="noisy_tensor",
        default = "random_col",
        metavar='string',
        choices=[
            'noisy_tensor'
            'random',
            'random_col',
            'mom_cons',
            'mom_cons_sv',
            'amino',
            'coil100',
            'timelapse',
            'scf',
            'embedding',
            'bert-param',
            'mm',
            'negrandom',
            'randn'
            ],
        help='choose tensor to test, available: noisy_tensor, random, negrandom,randn, random_col, mm, mom_cons, mom_cons_sv, amino, coil100, timelapse, scf (default: noisy_tensor)')
    parser.add_argument(
        '--tlib',
        default="numpy",
        metavar='string',
        choices=[
            'ctf',
            'numpy',
            ],
        help='choose tensor library teo test, choose between numpy and ctf (default: numpy)')
    parser.add_argument(
        '--method',
        default="DT",
        metavar='string',
        choices=[
            'DT',
            'DTLR',
            'PP',
            'partialPP',
            'NLS',
            'NLSALS',
            'SNLS'
            ],
        help='choose the optimization method: DT, PP, partialPP, DTLR (default: DT)')
    parser.add_argument(
        '--decomposition',
        default="CP",
        metavar='string',
        choices=[
            'CP',
            'Tucker',
            ],
        help='choose the decomposition method: CP, Tucker (default: CP)')
    parser.add_argument(
        '--hosvd',
        type=int,
        default=0,
        metavar='int',
        help='initialize factor matrices with hosvd or not (default: 0)')
    parser.add_argument(
        '--hosvd-core-dim',
        type=int,
        nargs='+',
        help='hosvd core dimensitionality.')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='int',
        help='random seed')
    parser.add_argument(
        '--tol',
        default=1e-5,
        type=float,
        metavar='float',
        help='Tolerance for stopping the iteration.')
    parser.add_argument(
        '--fit',
        default=0.99,
        type=float,
        metavar='float',
        help='Tolerance for stopping the iteration. default = 0.99')
    parser.add_argument(
        '--res-calc-freq',
        default=1,
        type=int,
        metavar='int',
        help='residual calculation frequency (default: 1).')
    parser.add_argument(
        '--save-tensor',
        action='store_true',
        help="Whether to save the tensor to file.")
    parser.add_argument(
        '--fast-residual',
        type=int,
        default=0,
        metavar= 'int',
        help='Enable fast residual calculation by using intermediates from the algorithm. Useful for large scale experiments for CPD (default: False)'
        )
    parser.add_argument(
        '--load-tensor',
        type=str,
        default='',
        metavar='str',
        help=
        'Where to load the tensor if the file exists. Empty means it starts from scratch. E.g. --load-tensor results/YOUR-FOLDER/ (do not forget the /)'
        )
    parser.add_argument(
        '--calc-cond',
        type=bool,
        default= False,
        metavar='bool',
        help='Calculate CPD condition number (default : False)')

def add_pp_arguments(parser):
    parser.add_argument(
        '--tol_restart_dt',
        default=0.01,
        type=float,
        metavar='float',
        help='used in pairwise perturbation optimizer, tolerance for dimention tree restart')

def add_col_arguments(parser):
    parser.add_argument(
        '--col',
        type=float,
        nargs='+',
        default=[0.75, 0.75],
        help='collinearity range')

def add_lrdt_arguments(parser):
    parser.add_argument(
        '--run-lowrank-dt',
        type=int,
        default=0,
        metavar='int',
        help='Run Dimension tree algorithm with low rank update on two of the factor matrices (default: 0)')
    parser.add_argument(
        '--do-lr-tol',
        default=1,
        type=int,
        metavar='int',
        help='Whether to perform low rank update by tolerance truncation.')
    parser.add_argument(
        '--lr-tol',
        default=0.1,
        type=float,
        metavar='float',
        help='Tolerance for low rank update truncation. This is the ratio of the singular values to be dropped. Can only be from 0 to 1.')
    parser.add_argument(
        '--sp-update-factor',
        type=int,
        default=0,
        metavar='int',
        help='use a sparse right factor in the low rank update scheme (default: 0)')
    parser.add_argument(
        '--num-lowr-init-iter',
        type=int,
        default=2,
        metavar='int',
        help='Number of initializing iterations (default: 2)')
    parser.add_argument(
        '--num-inter-iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of intermediate iterations when running low rand dimension tree with two fixed children of the root (default: 10)')

def add_general_arguments_3d(parser):
    parser.add_argument(
        '--run-naive',
        type=int,
        default=1,
        metavar='int',
        help='Run naive Dimension tree algorithm (default: 1)')
    parser.add_argument(
        '--run-lowrank',
        type=int,
        default=0,
        metavar='int',
        help='Run Dimension tree algorithm with low rank update (default: 0)')
    parser.add_argument(
        '--mm-test',
        type=int,
        default=0,
        metavar='int',
        help='decompose matrix multiplication tensor as opposed to random (default: 0)')
    parser.add_argument(
        '--pois-test',
        type=int,
        default=0,
        metavar='int',
        help='decompose Poisson tensor as opposed to random (default: 0)')
    parser.add_argument(
        '--num-slices',
        type=int,
        default=1,
        metavar='int',
        help='if greater than one do sliced standard ALS with this many slices (default: 1)')


def add_sparse_arguments(parser):
    parser.add_argument(
        '--sp-fraction',
        type=float,
        default=1.,
        metavar='float',
        help='sparsity (default: 1)')
    parser.add_argument(
        '--sp-updatelowrank',
        type=int,
        default=0,
        metavar='int',
        help='mem-preserving ordering of low-rank sparse contractions (default: 0)')
    parser.add_argument(
        '--sp',
        type=int,
        default=0,
        metavar='int',
        help='sparse decomposition (default: 0)')

def add_amdm_arguments(parser):
    parser.add_argument(
        '--thresh',
        type=float,
        default=None,
        metavar='float',
        help='Threshhold for ratio of inverting singular vals (default : None)')
    parser.add_argument(
        '--reduce-val',
        type=bool,
        default=False,
        metavar='bool',
        help='Reduce number of singular values to be inverted for hybrid algorithm (default : False)')
    parser.add_argument(
        '--reduce-val-freq',
        type=int,
        default=5,
        metavar='int',
        help='Reduce threhshold iteration frequency for hybrid algorithm (default : 5)')
    parser.add_argument(
        '--num_vals',
        type=int,
        default=None,
        metavar='int',
        help='Number of singular values inverted (default : None here but reset to rank in the file)')

def add_nls_arguments(parser):
    parser.add_argument(
    '--nls-tol',
    type=float,
    default= 1e-05,
    metavar='float',
    help='tolerance for nls to stop the iteration (default: 1e-05)')
    parser.add_argument(
    '--cg-tol',
    type=float,
    default=1e-03,
    metavar='float',
    help='tolerance for conjugate gradient method in nls (default: 1e-03)')
    parser.add_argument(
    '--grad-tol',
    type=float,
    default= 0.1,
    metavar='float',
    help='gradient tolerance for nls to stop the iteration (default: 0.1)')
    parser.add_argument(
    '--num',
    type=float,
    default=0,
    metavar='float',
    help='For controlling the step computed by GN (default:0)')
    parser.add_argument(
    '--switch-tol',
    type=float,
    default= 0.1,
    metavar='float',
    help='tolerance for switching to nls (default: 0.1)')
    parser.add_argument(
    '--own-cg',
    type=bool,
    default= False,
    metavar='bool',
    help='cg implementation for nls (default: False)')
    parser.add_argument(
    '--nls-iter',
    type=int,
    default= 2,
    metavar='int',
    help='Number of NLS iterations (default: 2)')
    parser.add_argument(
    '--als-iter',
    type=int,
    default= 30,
    metavar='int',
    help='Number of ALS iterations (default: 30)')
    parser.add_argument(
    '--maxiter',
    type= int,
    default= 0,
    metavar ='int',
    help ='Number of cg iterations for NLS (default: Nsr)')
    parser.add_argument(
    '--varying',
    type=int,
    default=1,
    metavar='int',
    help='varying regularization for nls (default: 1)')
    parser.add_argument(
    '--varying-fact',
    type= float,
    default= 2,
    metavar ='float',
    help ='Factor by which you would want to vary the regularization for NLS (default: 2)')
    parser.add_argument(
    '--lower',
    type= float,
    default=1e-06,
    metavar='float',
    help ='Lower Threshhold till we vary the regularization for NLS (default: 1e-06)')
    parser.add_argument(
    '--upper',
    type= float,
    default=1,
    metavar='float',
    help ='Upper Threshhold till we vary the regularization for NLS (default: 1)')
    parser.add_argument(
    '--diag',
    type=int,
    default=0,
    metavar='int',
    help='diagonal regularization for nls (default: 0 = False)')
    parser.add_argument(
    '--arm',
    type=int,
    default=0,
    metavar='int',
    help='Run with Armijo"s condition for line search (default: 0 = False)')
    parser.add_argument(
    '--c',
    type= float,
    default=1e-04,
    metavar='float',
    help ='Parameter for armijo"s line search (default: 1e-04)')
    parser.add_argument(
    '--tau',
    type= float,
    default=0.5,
    metavar='float',
    help ='Parameter for armijo"s line search (default: 0.5)')
    parser.add_argument(
    '--arm-iters',
    type=int,
    default=8,
    metavar='int',
    help='Max number of Armijo"s line search iterations (default: 8 )')
    parser.add_argument(
        '--experiment-prefix',
        '-ep',
        type=str,
        default='',
        required=False,
        metavar='str',
        help='Output csv file name prefix (default: None)')
    parser.add_argument(
        '--s',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in each dimension (default: 64)')
 ############################################################
    parser.add_argument(
        '--top_sin_val',
        type=int,
        default=1,
        metavar='int',
        help='Input the number of top singular values of empirical covariance (default: 3)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-4,
        metavar='float',
        help='Noise level (default: 1e-4)')
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        metavar='int',
        help='Input rank of noise tensor (default: 3)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.5,
        metavar='float',
        help='Input the rate of singular value decay (default: 1)')
    parser.add_argument(
        '--num_runs',
        type=int,
        default=5,
        metavar='int',
        help='Input number of runs (default: 5)')
    parser.add_argument(
        '--type-noisy-tensor',
        default="new_model",
        metavar='string',
        choices=[
            'old_model',
            'new_model'
            ],
        help='choose type of noisy tensor to test, available: old, new (default: new)')
    ############################################################
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        metavar='int',
        help='Tensor order (default: 3)')
    parser.add_argument(
        '--R',
        type=int,
        default=10,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--r',
        type=int,
        default=10,
        metavar='int',
        help='Update rank size (default: 10)')
    parser.add_argument(
        '--num-iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations (default: 10)')
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.0000001,
        metavar='float',
        help='regularization (default: 0.0000001)')
    parser.add_argument(
        '--tensor',
        default="noisy_tensor",
        metavar='string',
        choices=[
            'noisy_tensor'
            'random',
            'random_col',
            'mom_cons',
            'mom_cons_sv',
            'amino',
            'coil100',
            'timelapse',
            'scf',
            'embedding',
            'bert-param',
            'mm',
            'negrandom',
            'randn'
            ],
        help='choose tensor to test, available: noisy_tensor, random, negrandom,randn, random_col, mm, mom_cons, mom_cons_sv, amino, coil100, timelapse, scf (default: random)')
    parser.add_argument(
        '--tlib',
        default="ctf",
        metavar='string',
        choices=[
            'ctf',
            'numpy',
            ],
        help='choose tensor library teo test, choose between numpy and ctf (default: ctf)')
    parser.add_argument(
        '--method',
        default="NLS",
        metavar='string',
        choices=[
            'NLS',
            'NLSALS',
            'SNLS',
            'DT'
            ],
        help='choose the optimization method: DT, PP, partialPP, DTLR (default: DT)')
    parser.add_argument(
        '--decomposition',
        default="CP",
        metavar='string',
        choices=[
            'CP',
            'Tucker',
            ],
        help='choose the decomposition method: CP, Tucker (default: CP)')
    parser.add_argument(
        '--hosvd',
        type=int,
        default=0,
        metavar='int',
        help='initialize factor matrices with hosvd or not (default: 0)')
    parser.add_argument(
        '--hosvd-core-dim',
        type=int,
        nargs='+',
        help='hosvd core dimensitionality.')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='int',
        help='random seed')
    parser.add_argument(
        '--tol',
        default=1e-5,
        type=float,
        metavar='float',
        help='Tolerance for stopping the iteration.')
    parser.add_argument(
        '--res-calc-freq',
        default=1,
        type=int,
        metavar='int',
        help='residual calculation frequency (default: 1).')
    parser.add_argument(
        '--save-tensor',
        action='store_true',
        help="Whether to save the tensor to file.")
    parser.add_argument(
        '--load-tensor',
        type=str,
        default='',
        metavar='str',
        help=
        'Where to load the tensor if the file exists. Empty means it starts from scratch. E.g. --load-tensor results/YOUR-FOLDER/ (do not forget the /)'
        )
    


def add_probability_arguments(parser):
    parser.add_argument(
    '--num-gen',
    type=int,
    default=10,
    metavar='int',
    help='number of problems generated (default:10)')
    parser.add_argument(
    '--num-init',
    type=int,
    default=5,
    metavar='int',
    help='number of initializations (default:5)')
    parser.add_argument(
    '--conv-tol',
    type=float,
    default= 5e-05,
    metavar='float',
    help='tolerance for residual for convergence (default:5e-05)')
    parser.add_argument(
    '--f-R',
    type=int,
    default=3,
    metavar='int',
    help='First number for defining the range of Rank R (default:3)')
    parser.add_argument(
    '--l-R',
    type=int,
    default=6,
    metavar='int',
    help='Last number (including) for defining the range of R (default:6)')
    parser.add_argument(
        '--probmethod',
        default="DT",
        metavar='string',
        choices=[
            'DT',
            'DTLR',
            'PP',
            'partialPP',
            'NLS',
            'NLSALS',
            'SNLS'
            ],
    help='choose the optimization method: DT, PP, partialPP, DTLR (default: DT)')

def get_prob_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix,
            args.decomposition,
            args.probmethod,
            args.tensor,
            's' + str(args.s),
            'fR' + str(args.f_R),
            'lR'+ str(args.l_R),
            #'spfrac' + str(args.sp_fraction),
            #'splowrank' + str(args.sp_updatelowrank),
            #'runlowrank' + str(args.run_lowrank),
            #'runlowrankdt' + str(args.run_lowrank_dt),
            #'numinteriter' + str(args.num_inter_iter),
            #'pois' + str(args.pois_test),
            #'numslices' + str(args.num_slices),
            #'numinit-iter' + str(args.num_lowr_init_iter),
            'regu' + str(args.regularization),
            'tlib' + str(args.tlib)
        ]))
        
def get_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix,
            args.decomposition,
            args.method,
            args.tensor,
            's' + str(args.s),
            'R' + str(args.R),
            'r' + str(args.r),
            'k' + str(args.k),
            'epsilon' + str(args.epsilon),
            #'num_iter' + str(args.num_iter),
            #'top_sin_val' + str(args.top_sin_val),
            #'r' + str(args.r),
            #'spfrac' + str(args.sp_fraction),
            #'splowrank' + str(args.sp_updatelowrank),
            #'runlowrank' + str(args.run_lowrank),
            #'runlowrankdt' + str(args.run_lowrank_dt),
            #'numinteriter' + str(args.num_inter_iter),
            #'pois' + str(args.pois_test),
            #'numslices' + str(args.num_slices),
            #'numinit-iter' + str(args.num_lowr_init_iter),
            'alpha' + str(args.alpha),
            'num_runs' + str(args.num_runs),
            'regu' + str(args.regularization),
            'tlib' + str(args.tlib)
            #,'num_vals', + str(args.num_vals)
        ]))
