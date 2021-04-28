import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from mettack import MetaApprox, Metattack
from utils import *
from dataset import Dataset
import argparse
import scipy.sparse as sp
import os


def initial():
    # argument initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train.')  # origin = 200
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='citeseer',
                        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')  # origin = 0.05
    parser.add_argument('--model', type=str, default='Meta-Self',
                        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)
    return args, device


def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)  # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return loss_test.item(), acc_test.item()


def attack(adj, i=1):
    global features, labels, device, idx_train, lambda_, idx_unlabeled, perturbations
    # surrogate
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)

    # initialize attack
    if 'Self' in args.model:
        lambda_ = 0
    if 'Train' in args.model:
        lambda_ = 1
    if 'Both' in args.model:
        lambda_ = 0.5
    if 'A' in args.model:
        model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                           attack_features=False, device=device, lambda_=lambda_)
    else:
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                          attack_features=False, device=device, lambda_=lambda_)

    model = model.to(device)
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)

    mod_adj = model.modified_adj
    # modified_features = model.modified_features

    model.save_adj(root='./adj/', name='mod_adj' + str(i))
    # model.save_features(root='./fea/', name='mod_features'+i)
    return mod_adj


if __name__ == '__main__':

    # gpu test
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 0 if gpu, 1 if cpu
    print('gpu available:', torch.cuda.is_available())

    # gpu_count = torch.cuda.device_count()
    # print("gpu_count=",gpu_count)
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.current_device())

    args, device = initial()

    # load origin data
    data = Dataset(root='', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    perturbations = int(args.ptb_rate * (adj.sum() // 2))
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    loss_list = []
    acc_list = []
    k = 10

    loss_, acc_ = test(adj)
    loss_list.append(loss_)
    acc_list.append(acc_)

    for i in range(1, k+1):
        adj = attack(adj, i)
        loss_0, acc_0 = test(adj)
        loss_1, acc_1 = test(adj)
        loss_2, acc_2 = test(adj)
        loss_list.append((loss_0+loss_1+loss_2)/3)
        acc_list.append((acc_0+acc_1+acc_2)/3)

    print('loss:', loss_list)
    print('acc:', acc_list)
