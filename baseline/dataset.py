from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit, WikiCS
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily
from os import path
from torch_sparse import SparseTensor

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
    # elif args.dataset in 'arxiv':
    #     dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir)
    elif args.dataset in 'product':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_products_dataset(args.data_dir)

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in ('cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics', 'wikics', 'arxiv'):
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


def load_arxiv_dataset(data_dir, time_bound=[2015, 2017], inductive=False):
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

def load_proteins_dataset(data_dir, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    edge_feat = torch.as_tensor(ogb_dataset.graph['edge_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph['num_nodes'])
    node_feat = edge_index_.mean(dim=1)

    node_species = torch.as_tensor(ogb_dataset.graph['node_species'])
    species = [0] + node_species.unique().tolist()
    ind_species_min, ind_species_max = species[0], species[3]
    ood_tr_species_min, ood_tr_species_max = species[3], species[5]
    ood_te_species = [species[i] for i in range(5, 8)]

    center_node_mask = (node_species <= ind_species_max).squeeze(1) * (node_species > ind_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ind_species_max).squeeze(1)
        ind_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (node_species <= ood_tr_species_max).squeeze(1) * (node_species > ood_tr_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ood_tr_species_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in ood_te_species:
        center_node_mask = (node_species == i).squeeze(1)
        dataset = Data(x=node_feat, edge_index=edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_products_dataset(data_dir, inductive=False):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1)
    
    unique_labels = torch.unique(label)
    sorted_labels = torch.sort(unique_labels)[0]
    
    ood_te_classes = sorted_labels[-10:]
    ood_tr_class = sorted_labels[-11]
    # ind_classes = sorted_labels[:-11]
    ind_classes = sorted_labels[:-10]

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
    elif dataname == 'wikics':
        torch_dataset = WikiCS(root=f'{data_dir}WikiCS')
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask[:,0]]  # 使用第一个split
        tensor_split_idx['valid'] = idx[dataset.val_mask[:,0]]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    elif dataname == 'arxiv':
        from ogb.nodeproppred import NodePropPredDataset
        # print(f'{data_dir}ogb')
        # exit()
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
        elif dataname == 'wikics':
            class_t = 3
        label = dataset.y

        max_label = max(dataset.y)

        center_node_mask_ind = (label >= class_t)
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed', 'wikics'):
            split_idx = dataset.splits
        elif dataname == 'arxiv':
            split_idx = ogb_dataset.get_idx_split()
        if dataname in ('cora', 'citeseer', 'pubmed', 'arxiv', 'wikics'):
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

# def load_arxiv_dataset(data_dir, class_range=[0,30], graph_partial=True):
#     from ogb.nodeproppred import NodePropPredDataset
#
#     ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
#     edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
#     x = torch.as_tensor(ogb_dataset.graph['node_feat'])
#     label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
#
#     class_min, class_max = class_range[0], class_range[1]
#     center_node_mask = (label < class_max).squeeze(1) * (label >= class_min).squeeze(1)
#     if graph_partial:
#         all_node_mask = (label < class_max).squeeze(1)
#         edge_index, _ = subgraph(all_node_mask, edge_index)
#
#     split_idx = ogb_dataset.get_idx_split()
#     tensor_split_idx = {}
#     idx = torch.arange(label.size(0))
#     for key in split_idx:
#         mask = torch.zeros(label.size(0), dtype=torch.bool)
#         mask[torch.as_tensor(split_idx[key])] = True
#         tensor_split_idx[key] = idx[mask * center_node_mask]
#
#     dataset = Data(x=x, edge_index=edge_index, y=label)
#     dataset.splits = tensor_split_idx
#     dataset.node_idx = idx[center_node_mask]
#
#     return dataset




# def load_ppi_dataset(data_dir, graph_idx):
#     transform = T.NormalizeFeatures()
#     torch_dataset = PPI(root=f'{data_dir}PPI',
#                               split='train', transform=transform)
#     dataset = torch_dataset[int(graph_idx)]
#     dataset.node_idx = torch.arange(dataset.num_nodes)
#
#     return dataset


#
# def load_amazon_dataset(data_dir, name):
#     transform = T.NormalizeFeatures()
#     if name == 'amazon-photo':
#         torch_dataset = Amazon(root=f'{data_dir}Amazon',
#                                name='Photo', transform=transform)
#     elif name == 'amazon-computer':
#         torch_dataset = Amazon(root=f'{data_dir}Amazon',
#                                name='Computers', transform=transform)
#     dataset = torch_dataset[0]
#
#     return dataset
#
#
# def load_coauthor_dataset(data_dir, name):
#     transform = T.NormalizeFeatures()
#     if name == 'coauthor-cs':
#         torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
#                                  name='CS', transform=transform)
#     elif name == 'coauthor-physics':
#         torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
#                                  name='Physics', transform=transform)
#     dataset = torch_dataset[0]
#
#     return dataset
#
#
# def load_ogb_dataset(data_dir, name):
#     from ogb.nodeproppred import NodePropPredDataset
#
#     ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
#     edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
#     x = torch.as_tensor(ogb_dataset.graph['node_feat'])
#     label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
#
#     def ogb_idx_to_tensor():
#         split_idx = ogb_dataset.get_idx_split()
#         tensor_split_idx = {key: torch.as_tensor(
#             split_idx[key]) for key in split_idx}
#         return tensor_split_idx
#
#     dataset = Data(x=x, edge_index=edge_index, y=label)
#     dataset.load_fixed_splits = ogb_idx_to_tensor
#     return dataset