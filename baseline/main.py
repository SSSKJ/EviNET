import faulthandler
faulthandler.enable()

import torch
torch.set_printoptions(profile="full")

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import ShaDowKHopSampler

from logger import Logger_classify, Logger_detect, SimpleLogger
from data_utils import normalize, gen_normalized_adjs, evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters
from dataset import load_dataset
from parse import parser_add_main_args
from models import *

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
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1:
    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

# get the splits for all runs
if args.dataset in ['cora', 'citeseer', 'pubmed', 'wikics']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

# infer the number of classes for non one-hot and one-hot labels
# c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
# d = dataset_ind.x.shape[1]

# print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
#       + f"classes {c} | feats {d}")
# print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
# if isinstance(dataset_ood_te, list):
#     for i, data in enumerate(dataset_ood_te):
#         print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
# else:
#     print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

c = dataset_ind.y[[dataset_ind.node_idx]].max().item() + 1
d = dataset_ind.x.shape[1]

print(f'dataset_ind: {torch.unique(dataset_ind.y[dataset_ind.node_idx], return_counts=True)}')
idx = dataset_ind.splits['train']
print(f'dataset_ind train: {torch.unique(dataset_ind.y[idx], return_counts=True)}')
idx = dataset_ind.splits['valid']
print(f'dataset_ind valid: {torch.unique(dataset_ind.y[idx], return_counts=True)}')
idx = dataset_ind.splits['test']
print(f'dataset_ind test: {torch.unique(dataset_ind.y[idx], return_counts=True)}')

print(f'dataset_ood_tr: {torch.unique(dataset_ood_tr.y[dataset_ood_tr.node_idx], return_counts=True)}')
print(f'dataset_ood_te: {torch.unique(dataset_ood_te.y[dataset_ood_te.node_idx], return_counts=True)}')

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
    + f"classes {c} | feats {d}")
print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
if isinstance(dataset_ood_te, list):
    for i, data in enumerate(dataset_ood_te):
        print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
else:
    print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")


### metric for classification ###
if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

### logger for result report ###
if args.mode == 'classify':
    logger = Logger_classify(args.runs, args)
else:
    logger = Logger_detect(args.runs, args)


### Training loop ###
train_time = []
inference_time = []
odin_r = []
for run in range(args.runs):
    ### Load method ###
    if args.method == 'maxlogits':
        model = MaxLogits(d, c, args).to(device)
    elif args.method == 'doctor':
        model = Doctor(d, c, args).to(device)
    elif args.method == 'CRL':
        model = CRL(d, c, args).to(device)
    elif args.method == 'GNNSafe':
        model = GNNSafe(d, c, args).to(device)
    elif args.method == 'energymodel':
        model = EnergyModel(d, c, args).to(device)
    elif args.method == 'energyprop':
        model = EnergyProp(d, c, args).to(device)
    elif args.method == 'GPN':
        model = GPN(d, c, args).to(device)
    elif args.method == 'SGCN':
        teacher = MaxLogits(d, c, args).to(device)
        model = SGCN(d, c, args).to(device)
    elif args.method == 'GEBM':
        model = GEBM(d, c, args).to(device)
    elif args.method == 'OE':
        model = OE(d, c, args).to(device)
    elif args.method == "ODIN":
        model = ODIN(d, c, args).to(device)
    elif args.method == "Mahalanobis":
        model = Mahalanobis(d, c, args).to(device)

    if args.dataset in ('proteins', 'ppi'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    model.train()
    print('MODEL:', model)
    model.reset_parameters()
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.method == 'GPN':
        optimizer, _ = model.get_optimizer(lr=args.lr, weight_decay=args.weight_decay)
        warmup_optimizer = model.get_warmup_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')


    if args.method == 'SGCN':
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            teacher.train()
            # for d_in, d_out in zip(train_loader_ind, train_loader_ood):
            teacher_optimizer.zero_grad()
            teacher_loss = teacher.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
            teacher_loss.backward()
            teacher_optimizer.step()
        model.create_storage(dataset_ind, teacher, device)

    for epoch in range(args.epochs):
        model.train()
        # for d_in, d_out in zip(train_loader_ind, train_loader_ood):

        if args.method == 'GPN' and epoch < args.GPN_warmup:
            warmup_optimizer.zero_grad()
            tick = time.time()
            loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
            loss.backward()
            warmup_optimizer.step()
            tock = time.time()
            train_time.append(tock - tick)
        elif args.method == 'CRL':
            optimizer.zero_grad()
            tick = time.time()
            loss, idx, correct, output  = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
            loss.backward()
            optimizer.step()
            model.correctness_history.correctness_update(idx, correct, output)
            model.correctness_history.max_correctness_update(epoch)
            tock = time.time()
            train_time.append(tock - tick)
        else:
            optimizer.zero_grad()
            tick = time.time()
            loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
            loss.backward()
            optimizer.step()
            tock = time.time()
            train_time.append(tock - tick)


        if args.mode == 'classify':
            tick = time.time()
            result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
            tock = time.time()
            inference_time.append(tock - tick)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')
        else:
            if args.method == "ODIN" or args.method == 'Mahalanobis':
                if epoch == args.epochs - 1:
                    tick = time.time()
                    result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
                    tock = time.time()
                    inference_time.append(tock - tick)
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'AUROC: {100 * result[0]:.2f}%, '
                          f'AURC: {100 * result[1]*10:.2f}%, '
                          f'FPR95: {100 * result[2]:.2f}%, '
                          f'Test Score: {100 * result[-2]:.2f}%')
                    odin_r.append({'AUROC': result[0], 'AURC': result[1], 'FPR95': result[2], 'AUPR': result[3], 'MisD AUROC': result[4], 'MisD AUPR': result[5], 'Acc': result[6]})
                else:
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, ')
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    tick = time.time()
                    result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
                    tock = time.time()
                    inference_time.append(tock - tick)
                # print(result)
                logger.add_result(run, result)

                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'AUROC: {100 * result[0]:.2f}%, '
                          f'AURC: {100 * result[1]*10:.2f}%, '
                          f'FPR95: {100 * result[2]:.2f}%, '
                          f'Test Score: {100 * result[-2]:.2f}%')

        torch.cuda.empty_cache()
    if args.method != "ODIN" and args.method != "Mahalanobis":
        logger.print_statistics(run)

if args.method != "ODIN" and args.method != "Mahalanobis":
    results = logger.print_statistics()
else:
    results = torch.tensor([list(r.values()) for r in odin_r], dtype=torch.float) * 100

### Save results ###
import os

if not os.path.exists(f'results/{args.dataset}'):
    os.makedirs(f'results/{args.dataset}')
if args.method == 'ODIN' or args.method == 'Mahalanobis':
    filename = f'results/{args.dataset}/{args.method}_{args.prefix}.csv'
else:
    filename = f'results/{args.dataset}/{args.method}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    if args.method == 'ODIN' or args.method == 'Mahalanobis':
        write_obj.write(f"{args.backbone} {args.T} {args.noise} {args.lamda}\n")
    else:
        write_obj.write(f"{args.backbone} {args.m_in} {args.m_out} {args.lamda}\n")
    for k in range(results.shape[1] // 6):
        r = results[:, k * 3]
        write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 1]
        write_obj.write(f'OOD Test {k + 1} Final AURC: {r.mean()*10:.2f} ± {r.std()*10:.2f} ')
        r = results[:, k * 3 + 2]
        write_obj.write(f'OOD Test {k + 1} Final FPR95: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 3]
        write_obj.write(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 4]
        write_obj.write(f'OOD Test {k + 1} Final MisD AUROC: {r.mean():.2f} ± {r.std():.2f} ')
        r = results[:, k * 3 + 5]
        write_obj.write(f'OOD Test {k + 1} Final MisD AUPR: {r.mean():.2f} ± {r.std():.2f}\n')
    r = results[:, -1]
    write_obj.write(f'Best Test Score In Test Score: {r.mean():.2f} ± {r.std():.2f}\n')