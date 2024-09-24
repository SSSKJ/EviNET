import faulthandler
faulthandler.enable()

import os
import argparse
import numpy as np
import random

from dataset import load_data
from parse import parser_add_main_args
from models import GraphModel
from data_utils import get_measures, eval_aurc, eval_acc
from utils.calculation_tools import get_score

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

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te, c, d = load_data(args)

name = 'edl'
if args.prefix != '':
    name = f'{name}_{args.prefix}'

name = f'{name}_h{args.hidden_channels}_l{args.lamda}_g{args.gamma}_lr{args.lr}_d{args.dropout}_a{args.aggr}'

# if os.path.exists(f'./pretrain/{args.dataset}/results{name}.pth'):
#     exit(1)

num_features = args.hidden_channels
W = c

results = []

for run in range(args.runs):
    
    encoder = GraphModel(d, c, args, mode = 2).to(device)

    best_current = -100000
    best_epoch = 0
    best_auroc = 0
    best_auroc_d = 0
    best_aurc = 0
    best_fpr = 0
    test_acc = 0
    best_encoder = None
    
    optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=args.lr, weight_decay=0.005
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    
    ## pretrain node embedding
    print('Start Training')

    eval_nodeE = {}

    for epo in range(args.epochs):

        encoder.train()
        optimizer.zero_grad()

        train_idx = dataset_ind.splits['train']

        evidence = encoder(dataset_ind, device)[train_idx]
        y = dataset_ind.y[train_idx].squeeze(1)

        alpha = evidence + 1
        y_onehot = one_hot(y, num_classes = c).squeeze(1).to(device)

        S = torch.sum(alpha, dim=1, keepdim=True)
        Dirichlet_loss = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True).sum()

        alphas_t = y_onehot + (1 - y_onehot) * alpha
        K = torch.Tensor([c]).to(device)
        KL_loss = torch.lgamma(alphas_t.sum(axis = 1)) - (torch.lgamma(K) + torch.lgamma(alphas_t).sum(axis = 1)) + ((alphas_t - 1) * (torch.special.digamma(alphas_t)-torch.special.digamma(alphas_t.sum(axis = 1)).reshape(-1, 1))).sum(axis = 1)
        KL_loss = KL_loss.sum()
        lambda_t = min(1, epo/args.epochs)

        loss = Dirichlet_loss
        #+ lambda_t * KL_loss

        if not args.silent:
            print(f'Run: {run}, Epoch: {epo}, Loss: {loss.item()}, KL_loss: {KL_loss}, Dirichlet_loss: {Dirichlet_loss.item()}')
            
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # print('--------------------------------------------------')
        
        ## eval node embedding
        encoder.eval()

        evidence = encoder(dataset_ind, device)
        # print(evidence[dataset_ind.splits['train']])
        # print(evidence[dataset_ind.splits['test']])
        # print(evidence[dataset_ood_te.node_idx])

        for tag in ['train', 'valid', 'test']:
            idx = dataset_ind.splits[tag]
            e = evidence[idx]
            label = dataset_ind.y[idx].squeeze(1)

            prediction = e.argmax(-1)

            if tag not in eval_nodeE:
                eval_nodeE[tag] = []

            # eval_nodeE[tag].append(sum(np.array(label) == prediction.detach().cpu().numpy()) / len(label))
            eval_nodeE[tag].append(eval_acc(label.unsqueeze(1), prediction.unsqueeze(1)))

        s = 'ACC: '
        
        for tag in eval_nodeE:

            s += f'{tag}: {eval_nodeE[tag][-1]}, '

        ## 计算unknown detection指标
        test_ind_score_conflict, test_ind_score_vacuity = get_score(evidence[dataset_ind.splits['test']], c, args)
        # print('------------------------------------------------------')
        # print(test_ind_score_vacuity.mean())

        # print(test_ind_score)
        _, test_ood_score_vacuity = get_score(evidence[dataset_ood_te.node_idx], c, args)
        # print(test_ood_score_vacuity.mean())

        pred = evidence.argmax(-1).cpu()

        correct = (pred[dataset_ind.splits['test']] == dataset_ind.y[dataset_ind.splits['test']].view(1, -1).squeeze(0))
        auroc, aupr, fpr, _ = get_measures(test_ind_score_vacuity.cpu().detach(), test_ood_score_vacuity.cpu().detach())
        aurc, _ = eval_aurc(correct, test_ind_score_conflict.cpu().detach())
        # print(test_ind_score_conflict[correct])
        # print(test_ind_score_conflict[~correct])

        result = {'auroc': auroc, 'aurc': aurc, 'fpr95': fpr}

        for tag in result:

            s += f'{tag}: {result[tag]}, '

            
        print(f'{s}')

        if eval_nodeE['test'][-1] + auroc - aurc*10 > best_current:

            best_auroc_d = 0

            best_auroc = auroc
            best_fpr = fpr
            best_aurc = aurc

            test_acc = eval_nodeE['test'][-1]

            best_current = eval_nodeE['test'][-1] + auroc - aurc*10
            best_epoch = epo
            best_encoder = encoder.state_dict()
            
    # print(f'Run {run}: best current test acc: {best_test}, best current val acc: {best_current}, before fine-tuning encoder val acc: {encoder_history_acc}, best encoder acc: {best_encoder_acc}, best distance: {best_distance}, best loss: {best_loss}, best distance-based auroc: {best_auroc_d}, best auroc: {best_auroc}, best aurc: {best_aurc}, best eaurc: {best_eaurc}, best epoch: {best_epoch}, best result: {best_result}')
    results.append({'run': run, 'test acc': test_acc, 'best aurc': best_aurc, 'best fpr95': best_fpr, 'best auroc': best_auroc, 'best distance-based auroc': best_auroc_d, 'epo': best_epoch})

os.makedirs(f'./result/{args.dataset}/{args.prefix}/', exist_ok = True)
torch.save(results, f'./result/{args.dataset}/{args.prefix}/{name}.pth')

os.makedirs(f'./models/{args.dataset}/{args.prefix}/', exist_ok = True)
torch.save(best_encoder, f'./models/{args.dataset}/{args.prefix}/{name}_encoder.pth')

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
    auroc_d.append(r['best distance-based auroc']*100)

print(f'------------------summary------------------')
print(f'Acc: {np.array(acc).mean():.2f} ± {np.array(acc).std():.2f}')
print(f'AURC: {np.array(aurc).mean():.2f} ± {np.array(aurc).std():.2f}')
print(f'fpr: {np.array(fpr).mean():.2f} ± {np.array(fpr).std():.2f}')
print(f'auroc: {np.array(auroc).mean():.2f} ± {np.array(auroc).std():.2f}')
# print(auroc)
print(f'auroc_d: {np.array(auroc_d).mean():.2f} ± {np.array(auroc_d).std():.2f}')