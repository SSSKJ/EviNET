import faulthandler
faulthandler.enable()

import os
import argparse
import numpy as np
import random

from parse import parser_add_main_args
from models import GraphModel, InvLinear, GCN
from utils.calculation_tools import cal_logit, edl_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn import init

from dataset import load_arxiv_dataset_samples, load_erdos_renyi_dataset_with_density

import time

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        init.ones_(module.weight)
        init.zeros_(module.bias)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load data ###
# dataset_ind = load_arxiv_dataset_samples(args.data_dir, args.number_nodes, inductive = args.inductive)
dataset_ind = load_erdos_renyi_dataset_with_density(args.number_nodes, args.edge_density, 64, 10)

# print(dataset_ind.node_idx)

c = dataset_ind.y[dataset_ind.node_idx].max().item() + 1
d = dataset_ind.x.shape[1]
print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.nnz()} | "
        + f"classes {c} | feats {d}")

name = f'complexity_analysis_{args.number_nodes}_{args.edge_density}'

num_features = args.hidden_channels
run = 0

print(f'start EVINET {args.number_nodes}')

deepset = InvLinear(num_features, args.u_hidden, num_features, bias=True, reduction=args.aggr, dropout = args.dropout).to(device)
encoder = GraphModel(dataset_ind.x.size(1), num_features, args).to(device)

optimizer1 = torch.optim.Adam(
        [{'params': encoder.parameters()},
        {'params': deepset.parameters()}],
        lr=args.lr, weight_decay=0.005
    )

Beta2E = [GCN(args.hidden_channels * 2, args.hidden_channels * 2, 1, args.b2e_layers, args.b2e_dropout).to(device) for _ in range(c+1)]

optimizer2 = torch.optim.Adam(
        [{'params': m.parameters()} for m in Beta2E],
        lr=args.b2e_lr, weight_decay=0.005
    )


## pretrain node embedding
print('Start Training')

for loop in range(1):

    # 开始记录时间和内存
    m1start_time = time.time()
    torch.cuda.reset_peak_memory_stats()  # 重置GPU内存峰值统计
    m1start_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 记录初始内存

    tstart_time = time.time()
    tstart_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 记录初始内存


    for epo in range(20):

        encoder.train()
        deepset.train()

        optimizer1.zero_grad()

        betaE = encoder(dataset_ind, device)
        node_alphas = betaE[:, :int(betaE.size(1)/2)]
        node_betas = betaE[:, int(betaE.size(1)/2):]
            
        train_idx = dataset_ind.node_idx

        x_alpha = node_alphas[train_idx]
        x_beta = node_betas[train_idx]
        y = dataset_ind.y[train_idx].squeeze(1)

        l_a = []
        l_b = []

        for class_idx in range(c):

            classN = deepset(torch.cat([x_alpha[y == class_idx], x_beta[y == class_idx]], dim=-1).unsqueeze(0)).squeeze(0)
            classN_alpha = classN[:int(len(classN)/2)]
            classN_beta = classN[int(len(classN)/2):]

            l_a.append(classN_alpha)
            l_b.append(classN_beta)

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        class_union = deepset(torch.cat([class_alphas, class_betas], dim=-1).unsqueeze(0)).squeeze(0)
        classOther_alpha = class_union[:int(len(class_union)/2)]
        classOther_beta = class_union[int(len(class_union)/2):]

        l_a.append(1.0/classOther_alpha.squeeze(0))
        l_b.append(1.0/classOther_beta.squeeze(0))

        y_onehot = one_hot(dataset_ind.y[train_idx], num_classes = c).squeeze(1).to(device) ## N*C
        neg_y_onehot = 1 - one_hot(dataset_ind.y[train_idx], num_classes = c).squeeze(1).to(device)

        node2class_logit = cal_logit(x_alpha.unsqueeze(1), x_beta.unsqueeze(1), class_alphas.unsqueeze(0), class_betas.unsqueeze(0), args.gamma, 1) ## N*C

        positive_node2class_loss = -(y_onehot * F.logsigmoid(node2class_logit)).sum()
        negative_node2class_loss = -(((neg_y_onehot * F.logsigmoid(-node2class_logit)).sum(-1)) / (c - 1)).sum()

        node2class_loss = (positive_node2class_loss + negative_node2class_loss)

        loss = node2class_loss

        loss.backward()
        optimizer1.step()

    # 记录结束时间和内存
    m1end_time = time.time() - m1start_time
    m1end_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 获取峰值内存使用

    eval_nodeE = {}
    encoder.eval()
    deepset.eval()

    # 开始记录时间和内存
    m2start_time = time.time()

    torch.cuda.reset_peak_memory_stats()  # 重置GPU内存峰值统计
    m2start_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 记录初始内存

    for epo in range(20):

        for m in Beta2E:
            m.train()

        optimizer2.zero_grad()

        betaE = encoder(dataset_ind, device)
        node_alphas = betaE[:, :int(betaE.size(1)/2)]
        node_betas = betaE[:, int(betaE.size(1)/2):]
            
        train_idx = dataset_ind.node_idx

        x_alpha = node_alphas[train_idx]
        x_beta = node_betas[train_idx]
        y = dataset_ind.y[train_idx].squeeze(1)

        l_a = []
        l_b = []

        for class_idx in range(c):

            classN = deepset(torch.cat([x_alpha[y == class_idx], x_beta[y == class_idx]], dim=-1).unsqueeze(0)).squeeze(0)
            classN_alpha = classN[:int(len(classN)/2)]
            classN_beta = classN[int(len(classN)/2):]

            l_a.append(classN_alpha)
            l_b.append(classN_beta)

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        class_union = deepset(torch.cat([class_alphas, class_betas], dim=-1).unsqueeze(0)).squeeze(0)
        classOther_alpha = class_union[:int(len(class_union)/2)]
        classOther_beta = class_union[int(len(class_union)/2):]

        l_a.append(1.0/classOther_alpha.squeeze(0))
        l_b.append(1.0/classOther_beta.squeeze(0))

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        class_cat = torch.concat([class_alphas, class_betas], dim = -1)
        node_cat = torch.concat([node_alphas, node_betas], dim = -1)

        evidence = []
        for class_index in range(len(class_cat)):
            x = torch.cat([node_cat, class_cat[class_index].unsqueeze(0).expand(node_cat.size(0), class_cat.size(-1))], dim = -1)
            evidence.append(Beta2E[class_index](x.to(device), dataset_ind.edge_index.to(device)))
            
        evidence = torch.concat(evidence, -1)

        alpha = evidence[:, :-1] + 1/c * evidence[:, -1].view(-1, 1)

        loss = edl_loss(torch.digamma, dataset_ind.y[train_idx], alpha[train_idx], device, c)
            
        loss.backward()
        optimizer2.step()

# 记录结束时间和内存
m2end_time = time.time() - m2start_time
m2end_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 获取峰值内存使用

result = [m2end_time, m2end_memory - m2start_memory]
print(f'M2: {result} for {args.number_nodes}')
os.makedirs(f'./result/{args.dataset}/complexity/', exist_ok = True)
torch.save(result, f'./result/{args.dataset}/complexity/M2_{name}.pth')


result = [m1end_time, m1end_memory - m1start_memory]
print(f'M1: {result} for {args.number_nodes}')
os.makedirs(f'./result/{args.dataset}/complexity/', exist_ok = True)
torch.save(result, f'./result/{args.dataset}/complexity/M1_{name}.pth')

tend_time = time.time() - tstart_time
tend_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 获取峰值内存使用

result = [tend_time, tend_memory - tstart_memory]
print(f'Total: {result} for {args.number_nodes}')

os.makedirs(f'./result/{args.dataset}/complexity/', exist_ok = True)
torch.save(result, f'./result/{args.dataset}/complexity/EviNET_{name}.pth')

