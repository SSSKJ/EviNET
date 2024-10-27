from models import *
from data_utils import normalize

def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='Amazon-photo')
    parser.add_argument('--ood_type', type=str, default='label', choices=['structure', 'label', 'feature'])
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--path', default='./results/images', type=str)
    parser.add_argument('--prefix', default='all', type=str)
    parser.add_argument('--fix', action='store_true')

    # model network
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    
    # BetaE
    parser.add_argument('--gamma', default=60, type=int)

    # training
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')

    parser.add_argument('--b2e_lr', default = 0.0005, type=float)
    parser.add_argument('--b2e_layers', default = 3, type=int)
    parser.add_argument('--b2e_dropout', default = 0.4, type=float)
    parser.add_argument('--u_hidden', default = 64, type=int)



    parser.add_argument('--silent', action='store_true', help='')

    


    parser.add_argument('--number_nodes', default=100, type=int)
    parser.add_argument('--edge_density', default=0.005, type=float)

