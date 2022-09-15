import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def neighbor_density(feature, T=0.05):
    feature = F.normalize(feature)
    mat = torch.matmul(feature, feature.t()) / T
    mask = torch.eye(mat.size(0), mat.size(0)).bool()
    mat.masked_fill_(mask, -1 / T)
    result = entropy(mat)
    return result
