def add_general_arguments(parser):

    parser.add_argument(
        '--experiment-prefix',
        '-ep',
        type=str,
        default='',
        required=False,
        metavar='str',
        help='Output csv file name prefix (default: None)')
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
        default="random",
        metavar='string',
        choices=[
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
            'negrandom'
            ],
        help='choose tensor to test, available: random, random_col, mm, mom_cons, mom_cons_sv, amino, coil100, timelapse, scf (default: random)')
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

def add_pp_arguments(parser):
    parser.add_argument(
        '--tol-restart-dt',
        default=0.01,
        type=float,
        metavar='float',
        help='used in pairwise perturbation optimizer, tolerance for dimention tree restart')

def add_col_arguments(parser):
    parser.add_argument(
        '--col',
        type=float,
        nargs='+',
        default=[0.2, 0.8],
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
        '--sp-res',
        type=int,
        default=0,
        metavar='int',
        help='TTTP-based sparse residual calculation (default: 0)')

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
    default=1e-04,
    metavar='float',
    help='tolerance for conjugate gradient method in nls (default: 1e-04)')
    parser.add_argument(
    '--grad-tol',
    type=float,
    default= 1e-05,
    metavar='float',
    help='gradient tolerance for nls to stop the iteration (default: 1e-05)')
    parser.add_argument(
    '--num',
    type=float,
    default=0,
    metavar='float',
    help='For controlling the last step tolerance for nls (default:0)')
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
    help='tolerance for residual for if the method has converged (default:5e-05)')
    parser.add_argument(
    '--f-R',
    type=int,
    default=3,
    metavar='int',
    help='First number for defining the range of R (default:3)')
    parser.add_argument(
    '--l-R',
    type=int,
    default=6,
    metavar='int',
    help='Last number (including) for defining the range of R (default:6)')

def get_prob_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix,
            args.decomposition,
            args.method,
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
