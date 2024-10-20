import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import init
import torch.nn.functional as F

import math
import numpy as np

class InvLinear(nn.Module):

    def __init__(self, in_features, hidden_channels, out_features, bias=True, dropout = 0.2):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.projector1 = nn.Linear(in_features = in_features, out_features = hidden_channels)
        self.projector2 = nn.Linear(in_features = hidden_channels, out_features = in_features)

        self.beta = nn.Parameter(torch.Tensor(self.hidden_channels,
                                              self.hidden_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.hidden_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        if self.projector1.bias is not None:
            init.zeros_(self.projector1.bias)

        if self.projector2.bias is not None:
            init.zeros_(self.projector2.bias)

        init.xavier_uniform_(self.beta)
        init.xavier_uniform_(self.projector1.weight)
        init.xavier_uniform_(self.projector2.weight)
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):

        X = self.projector1(X)
        X = F.dropout(X, self.dropout)
        N, M, _ = X.shape
        device = X.device
        y = torch.zeros(N, self.hidden_channels).to(device)
        if mask is None:
            mask = torch.ones(N, M).bool().to(device)      

        sizes = mask.float().sum(dim=1).unsqueeze(1)
        Z = X * mask.unsqueeze(2).float()
        y = (Z.sum(dim=1) @ self.beta)/sizes

        if self.bias is not None:
            y += self.bias

        y = F.dropout(y, self.dropout)
        y = self.projector2(y)
        y = F.dropout(y, self.dropout)

        return F.softplus(y)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)
    
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0, save_mem=True, use_bn=True, mode = 0):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.mode = mode

        self.last_bn = nn.BatchNorm1d(out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            init.ones_(bn.weight)
            init.zeros_(bn.bias)

        init.ones_(self.last_bn.weight)
        init.zeros_(self.last_bn.bias)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.last_bn(x)
        x = F.softplus(x)
        return x
    
class GraphModel(nn.Module):
    def __init__(self, d, c, args, mode = 0):
        super(GraphModel, self).__init__()
        self.encoder = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn,
                    mode=mode)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)
    

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                init.zeros_(lin.bias)

        for bn in self.bns:
            init.ones_(bn.weight)
            init.zeros_(bn.bias)

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.bns[i](x)
        x = self.lins[-1](x)
        x = F.softplus(x)
        return x