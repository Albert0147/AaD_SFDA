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

def test_and_nd(step, dataset_test, name, G, C):
    G.eval()
    C.eval()
    correct = 0
    size = 0
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t  = data[0], data[1]
            img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                             Variable(label_t.cuda(), volatile=True)
            feat = G(img_t)
            out_t = C(feat).cpu()
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data.cpu()).cpu().sum()
            k = label_t.data.size()[0]
            size += k
            if batch_idx == 0:
                label_all = label_t
                feat_all = feat
                pred_all = out_t
            else:
                pred_all = torch.cat([pred_all, out_t],0)
                feat_all = torch.cat([feat_all, feat],0)
                label_all = torch.cat([label_all, label_t],0)
    print(
        '\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, size,
            100. * correct / size))
    ## Accuracy
    close_p = 100. * float(correct) / float(size)
    #compute_variance(pred_all, label_all)
    #compute_variance(feat_all, label_all)
    ## Entropy
    ent_class = entropy(pred_all)

    ## Neighborhood Density
    pred_soft = F.softmax(pred_all)
    nd_soft = neighbor_density(pred_soft)

    ## Neighborhood Density without softmax
    nd_nosoft = neighbor_density(pred_all)

    output = [step, "closed", "acc %s"%float(close_p),
              "neighborhood density %s"%nd_soft.item(),
              "neighborhood density no soft%s" % nd_nosoft.item(),
              "entropy class %s"%ent_class.item()]

    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)
    return close_p, nd_soft.item(), nd_nosoft.item(), ent_class.item()
