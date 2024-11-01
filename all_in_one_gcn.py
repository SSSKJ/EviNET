import faulthandler
faulthandler.enable()

import os
import argparse
import numpy as np
import random

from dataset import load_data
from parse import parser_add_main_args
from models import GraphModel, InvLinear, GCN
from data_utils import get_measures, eval_aurc, eval_acc
from utils.calculation_tools import cal_logit, get_scoreNN_gcn, edl_loss

import torch
import torch.nn.functional as F
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

name = f'{args.prefix}_h{args.hidden_channels}_uh{args.u_hidden}_g{args.gamma}_lr{args.lr}_blr{args.b2e_lr}_d{args.dropout}_bd{args.b2e_dropout}_l{args.b2e_layers}'

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te, c, d = load_data(args)

num_features = args.hidden_channels

results = []

for run in range(args.runs):
    
    deepset = InvLinear(num_features, args.u_hidden, num_features, bias=True, dropout = args.dropout).to(device)
    encoder = GraphModel(dataset_ind.x.size(1), num_features, args).to(device)

    best_current = -100000
    best_auroc = 0
    best_aurc = 0
    best_fpr = 0
    test_acc = 0

    best_result = {'run': run, 'best_auroc': 0, 'best_aurc': 0, 'best_fpr': 0, 'best_current': -100000, 'test_acc': 0}
    
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

    for epo in range(args.epochs):

        encoder.train()
        deepset.train()

        optimizer1.zero_grad()

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

        if not args.silent:
            print(f'Run: {run}, Epoch: {epo}, Loss: {loss.item()}, positive_node2class_loss: {positive_node2class_loss.item()}, negative_node2class_loss: {negative_node2class_loss.item()}')
            
        loss.backward()
        optimizer1.step()

    eval_nodeE = {}
    encoder.eval()
    deepset.eval()

    for epo in range(args.epochs):

        for m in Beta2E:
            m.train()

        optimizer2.zero_grad()

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

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        class_cat = torch.concat([class_alphas, class_betas], dim = -1)
        node_cat = torch.concat([node_alphas, node_betas], dim = -1)

        evidence = []
        for class_index in range(len(class_cat)):
            x = torch.cat([node_cat, class_cat[class_index].unsqueeze(0).expand(node_cat.size(0), class_cat.size(-1))], dim = -1)
            evidence.append(Beta2E[class_index](x.to(device), dataset_ind.edge_index.to(device)))
            
        evidence = torch.concat(evidence, -1)
        
        if args.fix:
            evidence[:, -1] = c

        alpha = evidence[:, :-1] + 1/c * evidence[:, -1].view(-1, 1)

        loss = edl_loss(torch.digamma, dataset_ind.y[train_idx], alpha[train_idx], device, c)


        if not args.silent:
            print(f'Run: {run}, Epoch: {epo}, Loss: {loss.item()}')
            
        loss.backward()
        optimizer2.step()
        # scheduler.step(loss)

        # print('--------------------------------------------------')
        
        ## eval node embedding
        for m in Beta2E:
            m.eval()

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

        class_cat = torch.concat([class_alphas, class_betas], dim = -1)
        node_cat = torch.concat([node_alphas, node_betas], dim = -1)

        evidence = []
        for class_index in range(len(class_cat)):
            x = torch.cat([node_cat, class_cat[class_index].unsqueeze(0).expand(node_cat.size(0), class_cat.size(-1))], dim = -1)
            evidence.append(Beta2E[class_index](x.to(device), dataset_ind.edge_index.to(device)))

        evidence = torch.concat(evidence, -1)

        for tag in ['train', 'valid', 'test']:
            idx = dataset_ind.splits[tag]
            label = dataset_ind.y[idx].squeeze(1)

            prediction = evidence[idx].argmax(-1)

            if tag not in eval_nodeE:
                eval_nodeE[tag] = []

            eval_nodeE[tag].append(eval_acc(label.unsqueeze(1), prediction.unsqueeze(1)))

        class_union = deepset(torch.cat([class_alphas, class_betas], dim=-1).unsqueeze(0)).squeeze(0)
        classOther_alpha = class_union[:int(len(class_union)/2)]
        classOther_beta = class_union[int(len(class_union)/2):]

        l_a.append(1.0/classOther_alpha.squeeze(0))
        l_b.append(1.0/classOther_beta.squeeze(0))

        class_alphas = torch.stack(l_a, dim=0)
        class_betas = torch.stack(l_b, dim=0) ## C*F

        s = 'ACC: '
        
        for tag in eval_nodeE:

            s += f'{tag}: {eval_nodeE[tag][-1]}, '

        class_cat = torch.concat([class_alphas, class_betas], dim = -1)

        evidence = []
        for class_index in range(len(class_cat)):
            x = torch.cat([node_cat, class_cat[class_index].unsqueeze(0).expand(node_cat.size(0), class_cat.size(-1))], dim = -1)
            evidence.append(Beta2E[class_index](x.to(device), dataset_ind.edge_index.to(device)))

        evidence = torch.concat(evidence, -1)

        test_ind_score_conflict, test_ind_score_vacuity = get_scoreNN_gcn(evidence, dataset_ind.splits['test'], args)

        # print(test_ind_score)
        _, test_ood_score_vacuity = get_scoreNN_gcn(evidence, dataset_ood_te.node_idx, args)

        pred = evidence[idx, :-1].argmax(-1).cpu()

        correct = (pred == dataset_ind.y[dataset_ind.splits['test']].view(1, -1).squeeze(0))
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
filename = f'results/{args.dataset}/{args.method}_{args.prefix}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{args.hidden_channels} {args.u_hidden} {args.gamma} {args.lr} {args.b2e_lr} {args.dropout} {args.b2e_dropout} {args.b2e_layers} {args.epochs}\n")
    for k in range(results.shape[1] // 3):
        r = results[:, k * 3]
        write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 1]
        write_obj.write(f'OOD Test {k + 1} Final AURC: {r.mean()*10:.2f} ± {r.std()*10:.2f} ')
        r = results[:, k * 3 + 2]
        write_obj.write(f'OOD Test {k + 1} Final FPR95: {r.mean():.2f} ± {r.std():.2f}\n')
    r = results[:, -1]
    write_obj.write(f'In Test Score: {r.mean():.2f} ± {r.std():.2f}\n')