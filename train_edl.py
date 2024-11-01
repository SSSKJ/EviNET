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
from torch.nn.functional import one_hot

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

name = f'{args.prefix}_h{args.hidden_channels}_lr{args.lr}_d{args.dropout}'

num_features = args.hidden_channels
W = c

results = []

for run in range(args.runs):
    
    encoder = GraphModel(d, c, args).to(device)

    best_current = -100000
    best_auroc = 0
    best_aurc = 0
    best_fpr = 0
    test_acc = 0

    best_result = {'run': run, 'best_auroc': 0, 'best_aurc': 0, 'best_fpr': 0, 'best_current': -100000, 'test_acc': 0}
    
    optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=args.lr, weight_decay=0.005
        )
    
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

        # print('--------------------------------------------------')
        
        ## eval node embedding
        encoder.eval()

        evidence = encoder(dataset_ind, device)

        for tag in ['train', 'valid', 'test']:
            idx = dataset_ind.splits[tag]
            e = evidence[idx]
            label = dataset_ind.y[idx].squeeze(1)

            prediction = e.argmax(-1)

            if tag not in eval_nodeE:
                eval_nodeE[tag] = []

            eval_nodeE[tag].append(eval_acc(label.unsqueeze(1), prediction.unsqueeze(1)))

        s = 'ACC: '
        
        for tag in eval_nodeE:

            s += f'{tag}: {eval_nodeE[tag][-1]}, '

        test_ind_score_conflict, test_ind_score_vacuity = get_score(evidence[dataset_ind.splits['test']], c, args)

        _, test_ood_score_vacuity = get_score(evidence[dataset_ood_te.node_idx], c, args)

        pred = evidence.argmax(-1).cpu()

        correct = (pred[dataset_ind.splits['test']] == dataset_ind.y[dataset_ind.splits['test']].view(1, -1).squeeze(0))
        auroc, aupr, fpr, _ = get_measures(test_ind_score_vacuity.cpu().detach(), test_ood_score_vacuity.cpu().detach())
        aurc, _ = eval_aurc(correct, test_ind_score_conflict.cpu().detach())

        result = {'auroc': auroc, 'aurc': aurc, 'fpr95': fpr}

        for tag in result:

            s += f'{tag}: {result[tag]}, '

            
        print(f'{s}')

        if eval_nodeE['test'][-1] + auroc - aurc*10 > best_result['best_current']:

            best_result['best_auroc'] = auroc
            best_result['best_fpr'] = fpr
            best_result['best_aurc'] = aurc

            best_result['test_acc'] = eval_nodeE['test'][-1]

            best_result['best_current'] = eval_nodeE['test'][-1] + auroc - aurc*10
                
    results.append([best_result['best_auroc'], best_result['best_aurc'], best_result['best_fpr'], best_result['test_acc']])

results = torch.tensor(results, dtype=torch.float) * 100

### Save results ###
import os
if not os.path.exists(f'results/{args.dataset}'):
    os.makedirs(f'results/{args.dataset}')
filename = f'results/{args.dataset}/{args.prefix}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{args.hidden_channels} {args.lr} {args.dropout} {args.epochs}\n")
    
    for k in range(results.shape[1] // 3):
        r = results[:, k * 3]
        write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 1]
        write_obj.write(f'OOD Test {k + 1} Final AURC: {r.mean()*10:.2f} ± {r.std()*10:.2f} ')
        r = results[:, k * 3 + 2]
        write_obj.write(f'OOD Test {k + 1} Final FPR95: {r.mean():.2f} ± {r.std():.2f}\n')
    r = results[:, -1]
    write_obj.write(f'In Test Score: {r.mean():.2f} ± {r.std():.2f}\n')