from models import *
from data_utils import normalize

def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='label', choices=['structure', 'label', 'feature'],
                        help='only for cora/amazon/arxiv datasets')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=200)

    # model network
    parser.add_argument('--method', type=str, default='maxlogits')
    parser.add_argument('--backbone', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for sgc')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--m_in', type=float, default=-5)
    parser.add_argument('--m_out', type=float, default=-1)
    # parser.add_argument('--m', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--prop_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--GPN_detect_type', type=str, default='Epist', choices=['Alea', 'Epist', 'Epist_wo_Net'])
    parser.add_argument('--GPN_warmup', type=int, default=5)

    #param for ODIN and Mahalanobis
    parser.add_argument('--noise', type=float, default=0.)

    # training
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')


    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])

    # GKDE hyperparameter
    parser.add_argument('--gkde_seed', default=42, type=int)
    parser.add_argument('--gkde_dim_hidden', default=16, type=int)
    parser.add_argument('--gkde_dropout_prob', default=0.5, type=float)
    parser.add_argument('--gkde_use_kernel', default=1, type=int)
    parser.add_argument('--gkde_lambda_1', default=0.001, type=float)
    parser.add_argument('--gkde_teacher_training', default=1, type=int)
    parser.add_argument('--gkde_use_bayesian_dropout', default=0, type=int)
    parser.add_argument('--gkde_sample_method', default='log_evidence', type=str)
    parser.add_argument('--gkde_num_samples_dropout', default=10, type=int)
    parser.add_argument('--gkde_loss_reduction', default=None, type=str)

    ## CRL hyperparameter
    parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')

    # gnnsafe hyper
    parser.add_argument('--use_reg', action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')

    parser.add_argument('--prefix', default='all', type=str)


    # GraphEBMWrapper hyperparameters
    parser.add_argument('--gamma_correction', type=float, default=0.1,
                        help='Weight for Gaussian regularization (γ)')
    parser.add_argument('--num_diffusion_steps', type=int, default=2,
                        help='Number of diffusion steps (k)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'sum', 'max'],
                        help='Aggregation strategy for multiscale energy')
    parser.add_argument('--diffusion_type', type=str, default='label_propagation',
                        choices=['label_propagation', 'feature_propagation'],
                        help='Choose between structure-based or feature-based diffusion')

