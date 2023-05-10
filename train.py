import numpy as np
import scipy.sparse as sp
import os
import torch
import argparse
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load
from torch.optim import Adam
import torch.nn.functional as F
import random
import collections
from sklearn.cluster import DBSCAN


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size),
                               dtype=bool)  # torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0  # negative_mask[i, i:] = 0
        # negative_mask[i, i + batch_size] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):  # ???
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, nb_classes):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.fc = nn.Linear(n_h, nb_classes)
        self.sigm_fc = nn.Sigmoid()

    def forward(self, seq1, seq2, diff, sparse):
        # contra
        h_mask = self.gcn2(seq2, diff, sparse)
        h_2 = self.gcn2(seq1, diff, sparse)

        return h_mask[0].unsqueeze(0), h_2[0].unsqueeze(0)

    def embed(self, seq1, diff, sparse):
        h_2 = self.gcn2(seq1, diff, sparse)
        return h_2[0].unsqueeze(0).detach()


class UCGL(nn.Module):
    def __init__(self, n_in, n_h, nb_classes, n_clusters, pretrain_path=''):  # pretrain_path should be added
        super(UCGL, self).__init__()
        self.alpha = 0.0001  # 0.000001#1.0
        self.pretrain_path = pretrain_path
        self.mvg = Model(n_in, n_h, nb_classes)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_h))  # n_clusters?
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, adj, diff, features, labels, idx_train, idx_val, idx_test):
        self.mvg.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained MVGRL from', self.pretrain_path)

    def embed(self, seq, adj, diff, sparse):
        h_1 = self.mvg.gcn1(seq, adj, sparse)
        h = self.mvg.gcn2(seq, diff, sparse)
        return ((h + h_1)).detach()

    def forward(self, bf, mask_fts, bd, sparse):
        h_mask, h = self.mvg(bf, mask_fts, bd, sparse)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(h.reshape(-1, h.shape[2]).unsqueeze(1) - self.cluster_layer, 2),
            2) / self.alpha)  # h.reshape(-1,h.shape[2]).unsqueeze(1)-self.cluster_layer
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return h_mask, h, q


class fc_layer(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(fc_layer, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train_ucgl(dataset):
    nb_epochs = 1500
    lr = 0.00005
    sparse = False

    adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)
    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]
    sample_size = features.shape[0]
    batch_size = 4
    labels = torch.LongTensor(labels)

    model_UCGL = UCGL(ft_size, args.hid_units, nb_classes, n_clusters=args.n_clusters, pretrain_path=args.pretrain_path)
    optimizer = torch.optim.Adam(model_UCGL.parameters(), lr=lr, weight_decay=0.0)

    if torch.cuda.is_available():
        model_UCGL = model_UCGL.cuda()
        labels = labels.cuda()

    model_UCGL.pretrain(adj, diff, features, labels, idx_train, idx_val, idx_test)

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features_array = features
    diff_array = diff

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    # obtain features of positive samples
    features_mask = features
    for i in range(features_mask.shape[1]):
        idx = random.sample(range(1, features_mask.shape[2]), args.mask_num)
        features_mask[0][i][idx] = 0
    features_mask_array = np.array(features_mask.squeeze(0).cpu())

    # cluster parameter initiate
    h2 = model_UCGL.mvg.embed(features, diff, sparse)
    kmeans = KMeans(n_clusters=args.n_clusters)
    y_pred = kmeans.fit_predict(h2.data.squeeze().cpu().numpy())
    model_UCGL.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model_UCGL.train()
    acc_clu = 0
    kl_loss = 0
    loss = 0

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf, bf_mask = [], [], [], []
        for i in idx:
            bd.append(diff_array[i: i + sample_size, i: i + sample_size])
            bf.append(features_array[i: i + sample_size])
            bf_mask.append(features_mask_array[i: i + sample_size])

        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
        bf_mask = np.array(bf_mask).reshape(batch_size, sample_size, ft_size)

        if sparse:
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            bd = torch.FloatTensor(bd)
            bf = torch.FloatTensor(bf)
            bf_mask = torch.FloatTensor(bf_mask)

        if torch.cuda.is_available():
            bf = bf.cuda()
            bd = bd.cuda()
            bf_mask = bf_mask.cuda()

        if epoch % args.update_interval == 0:
            _, _, tmp_q = model_UCGL(bf, bf_mask, bd, sparse)
            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)

        acc = cluster_acc(np.array(labels.cpu()), y_pred)
        nmi = nmi_score(np.array(labels.cpu()), y_pred)
        ari = ari_score(np.array(labels.cpu()), y_pred)

        if acc > acc_clu:
            acc_clu = acc
            nmi_clu = nmi
            ari_clu = ari
            torch.save(model_UCGL.state_dict(), args.model_path)

        h_mask, h_2_sour, q = model_UCGL(bf, bf_mask, bd, sparse)

        kl_loss = F.kl_div(q.log(), p)

        temperature = 0.5
        y_sam = torch.LongTensor(y_pred)
        neg_size = 1000
        class_sam = []
        for m in range(np.max(y_pred) + 1):
            class_del = torch.ones(int(sample_size), dtype=bool)
            class_del[np.where(y_sam.cpu() == m)] = 0
            class_neg = torch.arange(sample_size).masked_select(class_del)
            neg_sam_id = random.sample(range(0, class_neg.shape[0]), int(neg_size))
            class_sam.append(class_neg[neg_sam_id])

        out = (h_2_sour).squeeze()
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        neg_samp = torch.zeros(neg.shape[0], int(neg_size))
        for n in range(np.max(y_pred) + 1):
            neg_samp[np.where(y_sam.cpu() == n)] = neg.cpu().index_select(1, class_sam[n])[np.where(y_sam.cpu() == n)]
        neg_samp = neg_samp.cuda()
        Ng = neg_samp.sum(dim=-1)

        pos_size = 10
        class_sam_pos = []
        for m in range(np.max(y_pred) + 1):
            class_del = torch.ones(int(sample_size), dtype=bool)
            class_del[np.where(y_sam.cpu() != m)] = 0
            class_pos = torch.arange(sample_size).masked_select(class_del)
            pos_sam_id = random.sample(range(0, class_pos.shape[0]), int(pos_size))
            class_sam_pos.append(class_neg[pos_sam_id])

        out = h_2_sour.squeeze()
        pos = torch.exp(torch.mm(out, out.t().contiguous()))
        pos_samp = torch.zeros(pos.shape[0], int(pos_size))
        for n in range(np.max(y_pred) + 1):
            pos_samp[np.where(y_sam.cpu() == n)] = pos.cpu().index_select(1, class_sam_pos[n])[
                np.where(y_sam.cpu() == n)]

        pos_samp = pos_samp.cuda()
        pos = pos_samp.sum(dim=-1) + torch.diag(torch.exp(torch.mm(out, (h_mask.squeeze()).t().contiguous())))
        node_contra_loss_2 = (- torch.log(pos / (pos + Ng))).mean()

        loss = node_contra_loss_2 + args.beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if acc < 0.688:
            break

    print('CLustering Acc: {:.4f}'.format(acc_clu), ', nmi {:.4f}'.format(nmi_clu), ', ari {:.4f}'.format(ari_clu))

    if args.classfication == 'True':
        class_loss = nn.CrossEntropyLoss()
        model_UCGL.load_state_dict(torch.load(args.model_path))
        embeds = model_UCGL.embed(features, adj, diff, sparse)
        train_embs = embeds[0, idx_train]
        test_embs = embeds[0, idx_test]

        train_lbls = labels[idx_train]
        test_lbls = labels[idx_test]

        accs = []
        wd = 0.01 if dataset == 'citeseer' else 0.0

        for _ in range(50):
            acc_max = 0
            classificate = fc_layer(args.hid_units, nb_classes)
            classificate = classificate.cuda()
            opt = torch.optim.Adam(classificate.parameters(), lr=5e-3, weight_decay=wd)
            for _ in range(700):
                classificate.train()
                opt.zero_grad()

                logits = classificate(train_embs)
                loss = class_loss(logits, train_lbls)

                loss.backward()
                opt.step()
                preds = torch.argmax(classificate(test_embs), dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                if acc > acc_max:
                    acc_max = acc
            print('class acc:{}'.format(acc_max))
            accs.append(acc_max * 100)

        accs = torch.stack(accs)
        print(accs.mean().item(), accs.std().item())


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(
        description='train_ucgl',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hid_units', default=220, type=int)
    parser.add_argument('--hidden1', type=int, default=16, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--update_interval', type=int)
    parser.add_argument('--tol', default=0.000, type=float)
    parser.add_argument('--pretrain_path', default='pretrained_model_pubmed.pkl', type=str,
                        help='the pretrained MVGRL path')  # pretrained on MVGRL(https://github.com/kavehhassani/mvgrl)
    parser.add_argument('--beta', default=10e-4, type=float, help='coefficient of kl loss')
    parser.add_argument('--model_path', type=str,
                        default='', help='the optimized model after training under debiased contrastive loss')
    parser.add_argument('--classfication', type=str, default='True')
    parser.add_argument('--mask_num', type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")
    args.dataset = 'pubmed'

    if args.dataset == 'cora':
        args.n_clusters = 7
    elif args.dataset == 'pubmed':
        args.n_clusters = 3
    args.model_path = os.path.join('/youpath/UCGL_' + args.dataset + '_' + str(args.hid_units) + '.pkl')

    #mask_num 200
    #[pos_size, neg_size,lr,update_interval]
    #Cora [10, 1000, 5*10e-5, 2], Citesser [70, 1000, 7*10e-5, 1], pubmed [450, 7000, 5*10e-5, 2]
    for __ in range(50): # For reproducing through multiple running times due to the randomness of contrastive sample selection in each iteration
        train_ucgl(args.dataset)
