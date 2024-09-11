from models import *
from data_utils import normalize

def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='label', choices=['structure', 'label', 'feature'],
                        help='only for cora/amazon/arxiv datasets')
    parser.add_argument('--data_dir', type=str, default='./data/')
    # parser.add_argument('--data_dir', type=str, default='/mnt/nas/dataset_share/GNN-common-datasets')
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
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--lamda', type=float, default=1.0)

    # training
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01) ## 0.001 for citeseer
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')


    parser.add_argument('--path', default='./results/images', type=str)
    parser.add_argument('--prefix', default='all', type=str)

    parser.add_argument('--gamma', default=60, type=int)

    parser.add_argument('--silent', action='store_true', help='')

    parser.add_argument('--load_encoder', action='store_true', help='')
    parser.add_argument('--aggr', default='sum', type=str)

    parser.add_argument('--inductive', action='store_true', help='inductive setting')
    parser.add_argument('--concat', action='store_true', help='vacuity setting')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--fix', action='store_true')

    parser.add_argument('--number_nodes', default=100, type=int)