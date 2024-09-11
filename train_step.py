import faulthandler
faulthandler.enable()

import os
import argparse
import numpy as np
import random

from dataset import load_data
from parse import parser_add_main_args
from models import Regularizer, GraphModel, InvLinear
from data_utils import get_measures, eval_aurc, eval_acc
from utils.calculation_tools import cal_logit, cal_distance, get_scoreNN, conflict_uncertainty
from utils.plot_tools import plot_lines

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch_geometric.nn import GCNConv
from torch.nn import init

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te, c, d = load_data(args)

name = f'{args.prefix}_h{args.hidden_channels}_g{args.gamma}_lr{args.lr}_d{args.dropout}_a{args.aggr}'

num_features = args.hidden_channels

if args.inductive:

    name += '_ind'

results = []

for run in range(args.runs):

    losses = {'Loss': [], 'positive_node2class_loss': [], 'negative_node2class_loss': []}
    
    deepset = InvLinear(num_features, 64, num_features, bias=True, reduction=args.aggr, dropout = args.dropout).to(device)
    encoder = GraphModel(dataset_ind.x.size(1), num_features, args).to(device)


    best_current = -100000
    best_epoch = 0
    best_auroc = 0
    best_auroc_d = 0
    best_aurc = 0
    best_fpr = 0
    test_acc = 0
    best_encoder = None
    best_deepset = None
    # best_projector = None
    
    optimizer = torch.optim.Adam(
            [{'params': encoder.parameters()},
            {'params': deepset.parameters()}],
            # {'params': projector.parameters()}],
            lr=args.lr, weight_decay=0.005
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    
    ## pretrain node embedding
    print('Start Training')

    eval_nodeE = {}

    for epo in range(args.epochs):

        encoder.train()
        deepset.train()

        optimizer.zero_grad()

        betaE = encoder(dataset_ind, device)
        node_alphas = betaE[:, :int(betaE.size(1)/2)]
        node_betas = betaE[:, int(betaE.size(1)/2):]
            
        train_idx = dataset_ind.splits['train']

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

        losses['Loss'].append(loss.item())
        losses['positive_node2class_loss'].append(positive_node2class_loss.item())
        losses['negative_node2class_loss'].append(negative_node2class_loss.item())

        if not args.silent:
            print(f'Run: {run}, Epoch: {epo}, Loss: {loss.item()}, positive_node2class_loss: {positive_node2class_loss.item()}, negative_node2class_loss: {negative_node2class_loss.item()}')
            
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # print('--------------------------------------------------')
        
        ## eval node embedding
        encoder.eval()
        deepset.eval()

        betaE = encoder(dataset_ind, device)
        node_alphas = betaE[:, :int(betaE.size(1)/2)]
        node_betas = betaE[:, int(betaE.size(1)/2):]

        train_idx = dataset_ind.splits['train']

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

        distance = cal_distance(node_alphas.unsqueeze(1), node_betas.unsqueeze(1), class_alphas.unsqueeze(0), class_betas.unsqueeze(0), 1)

        for tag in ['train', 'valid', 'test']:

            idx = dataset_ind.splits[tag]
            label = dataset_ind.y[idx].squeeze(1)

            prediction = distance[idx].argmin(-1)

            if tag not in eval_nodeE:
                eval_nodeE[tag] = []

            eval_nodeE[tag].append(eval_acc(label.unsqueeze(1), prediction.unsqueeze(1)))

        # ## 类的补集的交集
        class_union = deepset(torch.cat([class_alphas, class_betas], dim=-1).unsqueeze(0)).squeeze(0)
        classOther_alpha = class_union[:int(len(class_union)/2)]
        classOther_beta = class_union[int(len(class_union)/2):]

        l_a.append(1.0/classOther_alpha.squeeze(0))
        l_b.append(1.0/classOther_beta.squeeze(0))

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        distance = cal_distance(node_alphas.unsqueeze(1), node_betas.unsqueeze(1), class_alphas.unsqueeze(0), class_betas.unsqueeze(0), 1)

        s = 'ACC: '
        
        for tag in eval_nodeE:

            s += f'{tag}: {eval_nodeE[tag][-1]}, '

        test_ind_score_conflict = 1.0/(conflict_uncertainty(1.0/distance)[dataset_ind.splits['test']])
        test_ind_score_vacuity = distance[dataset_ind.splits['test']][:, -1]
        test_ood_score_vacuity = distance[dataset_ood_te.node_idx][:, -1]

        pred = distance[dataset_ind.splits['test']][:, :-1].argmin(-1).cpu()

        correct = (pred == dataset_ind.y[dataset_ind.splits['test']].view(1, -1).squeeze(0))
        auroc, aupr, fpr, _ = get_measures(test_ind_score_vacuity.cpu().detach(), test_ood_score_vacuity.cpu().detach())
        aurc, _ = eval_aurc(correct, test_ind_score_conflict.cpu().detach())

        result = {'auroc': auroc, 'aurc': aurc, 'fpr95': fpr}

        for tag in result:

            s += f'{tag}: {result[tag]}, '

            
        print(f'{s}')

        if eval_nodeE['test'][-1] > best_current:

            best_auroc_d = 0

            best_auroc = auroc
            best_fpr = fpr
            best_aurc = aurc

            test_acc = eval_nodeE['test'][-1]

            best_current = eval_nodeE['test'][-1]
            best_epoch = epo

            best_encoder = encoder.state_dict()
            best_deepset = deepset.state_dict()
            
    results.append({'run': run, 'test acc': test_acc, 'best aurc': best_aurc, 'best fpr95': best_fpr, 'best auroc': best_auroc, 'best distance-based auroc': best_auroc_d, 'epo': best_epoch})
    plot_lines(losses['Loss'], {k: v for k, v in losses.items() if k != 'Loss'}, filename = f'{name}_loss.jpg', path = f'./curves/{args.dataset}/')

os.makedirs(f'./result/{args.dataset}/{args.prefix}/', exist_ok = True)
torch.save(results, f'./result/{args.dataset}/{args.prefix}/{name}.pth')

os.makedirs(f'./models/{args.dataset}/{args.prefix}/', exist_ok = True)
torch.save(best_encoder, f'./models/{args.dataset}/{args.prefix}/{name}_encoder.pth')
torch.save(best_deepset, f'./models/{args.dataset}/{args.prefix}/{name}_deepset.pth')

for r in results:
    print('------------------------------------------------------')
    print(r)


acc = []
aurc = []
fpr = []
auroc = []
auroc_d = []
for r in results:

    acc.append(r['test acc']*100)
    aurc.append(r['best aurc']*1000)
    fpr.append(r['best fpr95']*100)
    auroc.append(r['best auroc']*100)

print(f'------------------summary------------------')
print(f'Acc: {np.array(acc).mean():.2f} ± {np.array(acc).std():.2f}')
print(f'AURC: {np.array(aurc).mean():.2f} ± {np.array(aurc).std():.2f}')
print(f'fpr: {np.array(fpr).mean():.2f} ± {np.array(fpr).std():.2f}')
print(f'auroc: {np.array(auroc).mean():.2f} ± {np.array(auroc).std():.2f}')