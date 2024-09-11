import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
    
def conflict_uncertainty(evidence, default=1e-7):
    
    bs = evidence/evidence.sum(-1).unsqueeze(1)
    bs = bs[:, :-1]
    conflict_values = conflict(bs)
    return conflict_values

def conflict(b):
    n = b.size(1)
    mask = torch.eye(n).to(b.device) == 0  # creates a mask to remove diagonal elements
    
    denominator = b.sum(dim=1, keepdim=True) - b
    
    bal_values = 1 - torch.abs(b.unsqueeze(2) - b.unsqueeze(1)) / (b.unsqueeze(2) + b.unsqueeze(1))
    bal_values.masked_fill_(~mask, 0)  # set diagonal elements to 0
    
    numerator = (b.unsqueeze(2) * bal_values).sum(dim=1)
    
    result = (b * numerator / denominator).sum(dim=1)
    
    return result

def cal_logit(a1, b1, a2, b2, gamma, t = 0):

    return gamma - cal_distance(a1, b1, a2, b2, t)

def cal_distance(a1, b1, a2, b2, t = 0):

    query_dist = torch.distributions.beta.Beta(a1, b1)
    entity_dist = torch.distributions.beta.Beta(a2, b2)

    if t == 0:
        out = torch.distributions.kl.kl_divergence(query_dist, entity_dist)

    else:
        out = torch.distributions.kl.kl_divergence(entity_dist, query_dist)

    out = torch.norm(out, p=1, dim=-1)

    return out

def cal_distance_raw(a1, b1, a2, b2, t = 0):

    query_dist = torch.distributions.beta.Beta(a1, b1)
    entity_dist = torch.distributions.beta.Beta(a2, b2)
    out = torch.distributions.kl.kl_divergence(entity_dist, query_dist)

    return out

from torch.nn.functional import one_hot
def edl_loss(func, y, alpha, device, nclass):

    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    y_onehot = one_hot(y, num_classes = nclass).squeeze(1)

    A = torch.sum(y_onehot * (func(S) - func(alpha)), dim=1, keepdim=True)

    return A.sum()

def loglikelihood_loss(y, alpha, device, nclass):

    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    y_onehot = one_hot(y, num_classes = nclass).squeeze(1)
    loglikelihood_err = torch.sum((y_onehot - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood.sum()

def get_score(evidence, W, args):

    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    belief = evidence/S

    u_conflict = conflict(belief)
    u_vacuity = W/S
    
    return 1/u_conflict.view(-1, 1), 1/u_vacuity.view(-1, 1)

def get_scoreNN(x_alpha, x_beta, class_alphas, class_betas, model, args):

    class_cat = torch.concat([class_alphas, class_betas], dim = -1)
    node_cat = torch.concat([x_alpha, x_beta], dim = -1)

    expanded_node_cat = node_cat.unsqueeze(1).expand(-1, class_cat.size(0), -1)

    # 将类别嵌入扩展到每个样本
    expanded_class_cat = class_cat.unsqueeze(0).expand(node_cat.size(0), -1, -1)

    # 结合样本嵌入和类别嵌入
    all_cat = torch.cat([expanded_node_cat, expanded_class_cat], dim=-1).view(-1, args.hidden_channels * 2)

    evidence = model(all_cat).view(node_cat.size(0), class_cat.size(0))
    if args.residual:
        evidence[:, -1] += (class_cat.size(0) - 1)

    elif args.fix:
        evidence[:, -1] = (class_cat.size(0) - 1)

    belief = evidence/evidence.sum(-1).unsqueeze(1)

    u_conflict = conflict_uncertainty(evidence)
    u_vacuity = belief[:, -1]
    
    return -u_conflict.view(-1, 1), -u_vacuity.view(-1, 1)

def get_scoreNN_gcn(e, index, args):

    evidence = e[index]

    belief = evidence/evidence.sum(-1).unsqueeze(1)

    u_conflict = conflict_uncertainty(evidence)
    u_vacuity = belief[:, -1]
    
    return -u_conflict.view(-1, 1), -u_vacuity.view(-1, 1)

def get_scoreNN_attn(x_alpha, x_beta, class_alphas, class_betas, model, args):

    class_cat = torch.concat([class_alphas, class_betas], dim = -1)
    node_cat = torch.concat([x_alpha, x_beta], dim = -1)

    expanded_node_cat = node_cat.unsqueeze(1).expand(-1, class_cat.size(0), -1)

    # 将类别嵌入扩展到每个样本
    expanded_class_cat = class_cat.unsqueeze(0).expand(node_cat.size(0), -1, -1)

    # 结合样本嵌入和类别嵌入
    all_cat = torch.cat([expanded_node_cat, expanded_class_cat], dim=-1)
    evidence = model(all_cat).view(node_cat.size(0), class_cat.size(0))
    if args.residual:
        evidence[:, -1] += (class_cat.size(0) - 1)

    elif args.fix:
        evidence[:, -1] = (class_cat.size(0) - 1)

    belief = evidence/evidence.sum(-1).unsqueeze(1)

    u_conflict = conflict_uncertainty(evidence)
    u_vacuity = belief[:, -1]
    
    return -u_conflict.view(-1, 1), -u_vacuity.view(-1, 1)


def get_scoreNN_distance(x_alpha, x_beta, class_alphas, class_betas, model, args):

    raw_distance = cal_distance_raw(x_alpha.unsqueeze(1), x_beta.unsqueeze(1), class_alphas.unsqueeze(0), class_betas.unsqueeze(0)).view(-1, x_alpha.size(-1))

    evidence = model(raw_distance).view(-1, class_alphas.size(0))

    belief = evidence/evidence.sum(-1).unsqueeze(1)

    u_conflict = conflict_uncertainty(evidence)
    u_vacuity = belief[:, -1]
    
    return -u_conflict.view(-1, 1), -u_vacuity.view(-1, 1)
