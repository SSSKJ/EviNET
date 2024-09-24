import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import init
import torch.nn.functional as F

import math
import numpy as np

# class Regularizer():
#     def __init__(self, base_add, min_val, max_val):
#         self.base_add = base_add
#         self.min_val = min_val
#         self.max_val = max_val

#     def __call__(self, entity_embedding):
#         return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class InvLinear(nn.Module):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """
    def __init__(self, in_features, hidden_channels, out_features, bias=True, reduction='mean', dropout = 0.2, mode = 0):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.mode = mode
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction
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
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """

        X = self.projector1(X)
        X = F.dropout(X, self.dropout)
        N, M, _ = X.shape
        device = X.device
        y = torch.zeros(N, self.hidden_channels).to(device)
        if mask is None:
            mask = torch.ones(N, M).bool().to(device)      

        # if self.reduction == 'mean':
        sizes = mask.float().sum(dim=1).unsqueeze(1)
        Z = X * mask.unsqueeze(2).float()
        y = (Z.sum(dim=1) @ self.beta)/sizes

        # elif self.reduction == 'sum':
        #     Z = X * mask.unsqueeze(2).float()
        #     y = Z.sum(dim=1) @ self.beta

        # elif self.reduction == 'max':
        #     Z = X.clone()
        #     Z[~mask] = float('-Inf')
        #     y = Z.max(dim=1)[0] @ self.beta

        # else:  # min
        #     Z = X.clone()
        #     Z[~mask] = float('Inf')
        #     y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        y = F.dropout(y, self.dropout)
        y = self.projector2(y)
        y = F.dropout(y, self.dropout)

        return F.softplus(y)
        
        # return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)
    
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0, save_mem=True, use_bn=True, mode = 0):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
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

# class BetaIntersection(nn.Module):
#     def __init__(self, dim):
#         super(BetaIntersection, self).__init__()
#         self.dim = dim
#         self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
#         self.layer2 = nn.Linear(2 * self.dim, self.dim)

#         nn.init.xavier_uniform_(self.layer1.weight)
#         nn.init.xavier_uniform_(self.layer2.weight)

#     def forward(self, alpha_embeddings, beta_embeddings):
#         all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
#         layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
#         attention = F.softmax(self.layer2(layer1_act), dim=-2) # (num_conj, batch_size, dim)

#         alpha_embedding = torch.sum(attention * alpha_embeddings, dim=-2)
#         beta_embedding = torch.sum(attention * beta_embeddings, dim=-2)

#         return alpha_embedding, beta_embedding
        
#     def mapper(self, sample_alpha_embeddings, sample_beta_embeddings, class_alpha_embeddings, class_beta_embeddings):
#         # 假设 sample_alpha_embeddings 的形状为 (batch_size, dim)
#         # 假设 class_alpha_embeddings 的形状为 (num_classes, dim)

#         # 将样本嵌入扩展到每个类别
#         expanded_sample_alpha = sample_alpha_embeddings.unsqueeze(1).expand(-1, class_alpha_embeddings.size(0), -1)
#         expanded_sample_beta = sample_beta_embeddings.unsqueeze(1).expand(-1, class_beta_embeddings.size(0), -1)

#         # 将类别嵌入扩展到每个样本
#         expanded_class_alpha = class_alpha_embeddings.unsqueeze(0).expand(sample_alpha_embeddings.size(0), -1, -1)
#         expanded_class_beta = class_beta_embeddings.unsqueeze(0).expand(sample_beta_embeddings.size(0), -1, -1)
        
#         # 结合样本嵌入和类别嵌入
#         all_alpha_embeddings = torch.cat([expanded_sample_alpha, expanded_class_alpha], dim=-1).view(sample_beta_embeddings.size(0), class_alpha_embeddings.size(0), 2, -1)
#         all_beta_embeddings = torch.cat([expanded_sample_beta, expanded_class_beta], dim=-1).view(sample_beta_embeddings.size(0), class_alpha_embeddings.size(0), 2, -1)

#         # 应用神经网络层
#         layer1_act = F.relu(self.layer1(torch.cat([all_alpha_embeddings, all_beta_embeddings], dim=-1)))
#         attention = F.softmax(self.layer2(layer1_act), dim=-2)  # 注意维度的变化

#         # 计算交集
#         alpha_embedding = torch.sum(attention * all_alpha_embeddings, dim=-2)
#         beta_embedding = torch.sum(attention * all_beta_embeddings, dim=-2)

#         return alpha_embedding, beta_embedding
    

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

class MLPWithAttention(nn.Module):
    """ MLP with Multi-head Attention and Residual Connections """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5, num_heads=4):
        super(MLPWithAttention, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.num_layers = num_layers
        
        # First layer
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Adding attention to capture relationships in hidden channels
        self.attn_layers.append(nn.MultiheadAttention(hidden_channels, num_heads=num_heads))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.attn_layers.append(nn.MultiheadAttention(hidden_channels, num_heads=num_heads))
        
        # Last layer
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        # Dropout rate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers
        for lin in self.lins:
            init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                init.zeros_(lin.bias)
        
        # Initialize batch normalization layers
        for bn in self.bns:
            init.ones_(bn.weight)
            init.zeros_(bn.bias)

    def forward(self, x, edge_index=None):
        batch_size, num_categories, _ = x.size()  # [397, 5, 32]
        residual = x  # For residual connection
        
        for i, lin in enumerate(self.lins[:-1]):
            # print(f"Layer {i} input shape: {x.shape}")
            
            # Linear layer
            x = lin(x)
            # print(f"Layer {i} linear output shape: {x.shape}")
            
            # Activation and dropout
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply batch normalization
            x = x.view(-1, x.size(-1))  # [397 * 5, hidden_channels]
            x = self.bns[i](x)
            x = x.view(batch_size, num_categories, -1)  # [397, 5, hidden_channels]
            # print(f"Layer {i} batch norm output shape: {x.shape}")
            
            # Apply multi-head attention
            x = x.transpose(0, 1)  # Attention expects input shape [num_categories, batch_size, embed_dim]
            attn_output, attn_weights = self.attn_layers[i](x, x, x)
            x = attn_output.transpose(0, 1)  # [batch_size, num_categories, hidden_channels]
            # print(f"Layer {i} attention output shape: {x.shape}")
            
            # Residual connection
            x = x + residual
            residual = x  # Update residual for next layer
        
        # Final layer without batch norm or attention
        x = self.lins[-1](x)  # [397, 5, 1]
        x = F.softplus(x)
        # print(f"Final layer output shape: {x.shape}")

        exit(1)
        
        return x
