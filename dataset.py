from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import even_quantile_labels, to_sparse_tensor, rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily
from torch_sparse import SparseTensor
from os import path

def load_data(args):

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
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        pass
    else:
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    # infer the number of classes for non one-hot and one-hot labels
    c = dataset_ind.y[dataset_ind.node_idx].max().item() + 1
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

    return dataset_ind, dataset_ood_tr, dataset_ood_te, c, d

def load_dataset(args):
    '''
    dataset_ind: in-distribution training dataset
    dataset_ood_tr: ood-distribution training dataset as ood exposure
    dataset_ood_te: a list of ood testing datasets or one ood testing dataset
    '''
    # multi-graph datasets, use one as ind, the other as ood
    if args.dataset == 'twitch':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_twitch_dataset(args.data_dir)

    # single graph, use partial nodes as ind, others as ood according to domain info
    elif args.dataset in 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir, inductive = args.inductive)
    elif args.dataset in 'product':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_products_dataset(args.data_dir, inductive = args.inductive)

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in ('cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_twitch_dataset(data_dir):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
                              name=subgraph_names[i], transform=transform)
        dataset = torch_dataset[0]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i == train_idx:
            dataset_ind = dataset
        elif i == valid_idx:
            dataset_ood_tr = dataset
        else:
            dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te

import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import networkx as nx
from torch_geometric.utils import from_networkx

def calculate_num_edges(num_nodes, edge_density):
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = int(edge_density * max_possible_edges)
    return num_edges

def load_erdos_renyi_dataset_with_density(num_nodes, edge_density, num_features, num_classes):
    num_edges = calculate_num_edges(num_nodes, edge_density)
    print(num_edges)

    G = nx.gnm_random_graph(n=num_nodes, m=num_edges)

    data = from_networkx(G)

    node_feat = torch.randn(num_nodes, num_features)

    labels = torch.randint(0, num_classes, (num_nodes,)).reshape(-1, 1)

    edge_index = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(num_nodes, num_nodes))

    node_idx = torch.arange(num_nodes)

    dataset = Data(x=node_feat, edge_index=edge_index, y=labels)
    dataset.node_idx = node_idx

    return dataset


def load_arxiv_dataset(data_dir, time_bound=[2015, 2017], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = torch.as_tensor(ogb_dataset.graph['node_year'])

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = torch.as_tensor((year <= year_max).squeeze(1), dtype=torch.bool)
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index
    ind_edge_index = SparseTensor(row=ind_edge_index[0], col=ind_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(1) & (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = center_node_mask
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index
    ood_tr_edge_index = SparseTensor(row=ood_tr_edge_index[0], col=ood_tr_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    center_node_mask = torch.zeros(year.size(0), dtype=torch.bool)
    for i in range(len(test_year_bound) - 1):
        mask = (year <= test_year_bound[i + 1]).squeeze(1) & (year > test_year_bound[i]).squeeze(1)
        center_node_mask |= mask

    if inductive:
        all_node_mask = center_node_mask
        ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_te_edge_index = edge_index
    ood_te_edge_index = SparseTensor(row=ood_te_edge_index[0], col=ood_te_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ood_te = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
    dataset_ood_te.node_idx = torch.arange(label.size(0))[center_node_mask]

    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_products_dataset(data_dir, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1)
    
    unique_labels = torch.unique(label)
    sorted_labels = torch.sort(unique_labels)[0]
    
    ood_te_classes = sorted_labels[-10:]
    ood_tr_class = sorted_labels[-11]
    ind_classes = sorted_labels[:-11]

    ind_mask = torch.isin(label, ind_classes)
    ood_tr_mask = label == ood_tr_class
    ood_te_mask = torch.isin(label, ood_te_classes)

    if inductive:
        ind_edge_index, _ = subgraph(ind_mask, edge_index)
    else:
        ind_edge_index = edge_index
    ind_edge_index = SparseTensor(row=ind_edge_index[0], col=ind_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    dataset_ind.node_idx = torch.arange(label.size(0))[ind_mask]

    if inductive:
        ood_tr_edge_index, _ = subgraph(ood_tr_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index
    ood_tr_edge_index = SparseTensor(row=ood_tr_edge_index[0], col=ood_tr_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    dataset_ood_tr.node_idx = torch.arange(label.size(0))[ood_tr_mask]

    if inductive:
        ood_te_edge_index, _ = subgraph(ood_te_mask, edge_index)
    else:
        ood_te_edge_index = edge_index
    ood_te_edge_index = SparseTensor(row=ood_te_edge_index[0], col=ood_te_edge_index[1], sparse_sizes=(node_feat.size(0), node_feat.size(0)))
    dataset_ood_te = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
    dataset_ood_te.node_idx = torch.arange(label.size(0))[ood_te_mask]

    return dataset_ind, dataset_ood_tr, dataset_ood_te

def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    n = data.num_nodes

    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    # h = homophily(data.edge_index, data.y)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset

def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset

def create_label_noise_dataset(data):

    y = data.y
    n = data.num_nodes
    idx = torch.randperm(n)[:int(n * 0.5)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max(), (int(n * 0.5), ))

    dataset = Data(x=data.x, edge_index=data.edge_index, y=y_new)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset

def load_graph_dataset(data_dir, dataname, ood_type):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'arxiv':
        from ogb.nodeproppred import NodePropPredDataset
        print(f'{data_dir}ogb')
        exit()
        ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}ogb')
        edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
        x = torch.as_tensor(ogb_dataset.graph['node_feat'])
        label = torch.as_tensor(ogb_dataset.labels).squeeze(1)
        dataset = Data(x=x, edge_index=edge_index, y=label)
    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'structure':
        dataset_ood_tr = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
        dataset_ood_te = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
    elif ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        if dataname == 'cora':
            class_t = 3
        elif dataname == 'amazon-photo':
            class_t = 4
        elif dataname == 'coauthor-cs':
            class_t = 4
        elif dataname == 'arxiv':
            class_t = 19
        elif dataname == 'coauthor-physics':
            class_t = 2
        elif dataname == 'amazon-computer':
            class_t = 3
        elif dataname == 'citeseer':
            class_t = 2
        label = dataset.y

        max_label = max(dataset.y)

        center_node_mask_ind = (label >= class_t)
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed'):
            split_idx = dataset.splits
        elif dataname == 'arxiv':
            split_idx = ogb_dataset.get_idx_split()
        if dataname in ('cora', 'citeseer', 'pubmed', 'arxiv'):
            tensor_split_idx = {}
            idx = torch.arange(label.size(0))
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask * center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        if dataname == 'arxiv':
            center_node_mask_ood_tr = (label <= class_t) * (label > class_t-5)
            center_node_mask_ood_te = (label <= class_t-5)
        else:
            center_node_mask_ood_tr = (label == class_t)
            center_node_mask_ood_te = (label < class_t)
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    else:
        raise NotImplementedError
    
    dataset_ind.y = max_label - dataset_ind.y
    dataset_ood_tr.y = max_label - dataset_ood_tr.y
    dataset_ood_te.y = max_label - dataset_ood_te.y

    return dataset_ind, dataset_ood_tr, dataset_ood_te