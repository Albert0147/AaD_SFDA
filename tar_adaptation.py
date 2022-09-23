import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, fea_bank, socre_bank, netF, netB, netC, args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    label_bank = torch.randn(num_sample)  # .cuda()
    pred_bank = torch.randn(num_sample)
    # nu=[]
    # s=[]
    # var_all=[]

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            """if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)"""

            # if args.singular:
            # _, ss, _ = torch.svd(fea)
            # s10=ss[:10]/ss[0]
            # s.append(s10)

            outputs = netC(fea)
            softmax_out = nn.Softmax()(outputs)
            nu.append(torch.mean(torch.svd(softmax_out)[1]))
            output_f_norm = F.normalize(fea)
            # fea_bank[indx] = output_f_norm.detach().clone().cpu()
            label_bank[indx] = labels.float().detach().clone()  # .cpu()
            pred_bank[indx] = outputs.max(-1)[1].float().detach().clone().cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                # all_fea = output_f_norm.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                # all_fea = torch.cat((all_fea, output_f_norm.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )

    _, socre_bank_ = torch.max(socre_bank, 1)
    distance = fea_bank.cpu() @ fea_bank.cpu().T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=4)
    score_near = socre_bank_[idx_near[:, :]].float().cpu()  # N x 4

    """acc1 = (score_near.mean(
        dim=-1) == score_near[:, 0]).sum().float() / score_near.shape[0]"""
    acc1 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == pred_bank)
    ).sum().float() / score_near.shape[0]
    acc2 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == label_bank)
    ).sum().float() / score_near.shape[0]

    """if True:
        nu_mean=sum(nu)/len(nu)"""

    # s10_avg=torch.stack(s).mean(0)
    # print('nuclear mean: {:.2f}'.format(nu_mean))

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        if True:
            return aacc, acc  # , acc1, acc2#, nu_mean, s10_avg

    else:
        return accuracy * 100, mean_ent


def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir_src + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_B.pt"
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{"params": v, "lr": args.lr * 1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)

            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0

    real_max_iter = max_iter

    while iter_num < real_max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product

        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == "visda-2017":
                acc, accc = cal_acc(
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                    "Task: {}, Iter:{}/{};  Acc on target: {:.2f}".format(
                        args.name, iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            netF.train()
            netB.train()
            netC.train()
            """if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.tag) + ".pt"))"""

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda-2017")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="weight/target/")
    parser.add_argument("--output_src", type=str, default="weight/source/")
    parser.add_argument("--tag", type=str, default="LPA")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=True)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")
    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "visda-2017":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = "./data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.da, args.dset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output,
            args.da,
            args.dset,
            names[args.s][0].upper() + names[args.t][0].upper(),
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
