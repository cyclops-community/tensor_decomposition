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
        default=0.1,
        metavar='float',
        help='regularization (default: 0.1)')
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


def get_file_prefix(args):
        return "-".join(filter(None, [
            args.experiment_prefix, 
            's' + str(args.s),
            'R' + str(args.R),
            'r' + str(args.r),
            'spfrac' + str(args.sp_fraction),
            'splowrank' + str(args.sp_updatelowrank),
            'runlowrank' + str(args.run_lowrank),
            'pois' + str(args.pois_test),
            'numslices' + str(args.num_slices),
            'numinit-iter' + str(args.num_lowr_init_iter),
            'regu' + str(args.regularization),
        ]))


