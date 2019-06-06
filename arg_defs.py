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
        '--num-lowr-init-iter',
        type=int,
        default=2,
        metavar='int',
        help='Number of initializing iterations (default: 2)')
    parser.add_argument(
        '--sp-fraction',
        type=float,
        default=1.,
        metavar='float',
        help='sparsity (default: 1)')
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.0000001,
        metavar='float',
        help='regularization (default: 0.0000001)')
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
        '--run-lowrank-dt',
        type=int,
        default=0,
        metavar='int',
        help='Run Dimension tree algorithm with low rank update on two of the factor matrices (default: 0)')
    parser.add_argument(
        '--num-inter-iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of intermediate iterations when running low rand dimension tree with two fixed children of the root (default: 10)')
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
    parser.add_argument(
        '--sp-update-factor',
        type=int,
        default=0,
        metavar='int',
        help='use a sparse right factor in the low rank update scheme (default: 0)')
    parser.add_argument(
        '--tensor',
        default="random",
        metavar='string',
        choices=[
            'random',
            'mom_cons',
            'mom_cons_sv',
            'amino',
            'coil100',
            'timelapse',
            ],
        help='choose tensor to test, available: random, mom_cons, mom_cons_sv, amino, coil100, timelapse (default: random)')
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
        '--lr-tol',
        default=0,
        type=float,
        metavar='float',
        help='Tolerance for low rank update truncation.')
    parser.add_argument(
        '--do-lr-tol',
        default=0,
        type=int,
        metavar='int',
        help='Whether to perform low rank update by tolerance truncation.')
    parser.add_argument(
        '--tol-restart-dt',
        default=0.01,
        type=float,
        metavar='float',
        help='used in pairwise perturbation optimizer, tolerance for dimention tree restart')




def get_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix,
            args.decomposition,
            args.method,
            's' + str(args.s),
            'R' + str(args.R),
            'r' + str(args.r),
            'spfrac' + str(args.sp_fraction),
            'splowrank' + str(args.sp_updatelowrank),
            'runlowrank' + str(args.run_lowrank),
            'runlowrankdt' + str(args.run_lowrank_dt),
            'numinteriter' + str(args.num_inter_iter),
            'pois' + str(args.pois_test),
            'numslices' + str(args.num_slices),
            'numinit-iter' + str(args.num_lowr_init_iter),
            'regu' + str(args.regularization),
        ]))
