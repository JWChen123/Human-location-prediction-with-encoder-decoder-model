# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch

import numpy as np
import pickle
import random
import torch.nn.functional as F
import scipy.sparse as sp
from collections import deque


class RnnParameterData(object):
    def __init__(self,
                 loc_emb_size=300,
                 hidden_size=300,
                 lr=1e-4,
                 lr_step=3,
                 lr_decay=0.5,
                 dropout_p=0.6,
                 L2=1e-5,
                 clip=3.0,
                 optim='Adam',
                 rnn_type='LSTM',
                 data_path='./data/',
                 save_path='./results/',
                 data_name='foursquare_2012'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(
            open(self.data_path + self.data_name + '.pk', 'rb'),
            encoding='iso-8859-1')
        self.vid_look_up = data['vid_lookup']
        self.vid_list = data['vid_list']
        self.data_neural = data['data_neural']

        self.loc_size = len(self.vid_list)
        self.loc_emb_size = loc_emb_size
        self.hidden_size = hidden_size

        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.rnn_type = rnn_type


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


# Heavily borrowed from https://github.com/spro/practical-pytorch.git
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = torch.LongTensor(length).cuda()
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


# simple seq2seq
def generate_input_long_history3(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            if len(target) == 1:
                pass

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(
                np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(
                np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = torch.LongTensor(loc_np)
            trace['tim'] = torch.LongTensor(tim_np)
            trace['target'] = torch.LongTensor(target)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


# batch_preparation
def generate_batch_long_history(data_neural, mode, candidate=None):
    pairs = []
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        for c, i in enumerate(train_id):
            sub_pair = []
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            decoder_input = [s[0] for s in session[:-1]]

            target = [s[0] for s in session[1:]]
            if len(target) == 1:
                pass
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            loc_tim = history
            loc_tim.append(session[0])
            loc_np = [s[0] for s in loc_tim]
            sub_pair.append(loc_np)
            sub_pair.append(decoder_input)
            sub_pair.append(target)
            pairs.append(sub_pair)
    return pairs


def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq


# batch_test
def random_batch(batch_size, pairs):
    history_seqs = []
    current_seqs = []
    target_seqs = []
    for _ in range(batch_size):
        pair = random.choice(pairs)
        history_seqs.append(pair[0])
        current_seqs.append(pair[1])
        target_seqs.append(pair[2])
    seq_pairs = sorted(
        zip(history_seqs, current_seqs, target_seqs),
        key=lambda p: len(p[0]),
        reverse=True)
    history_seqs, current_seqs, target_seqs = zip(*seq_pairs)
    history_lengths = [len(t) for t in history_seqs]
    history_padded = [pad_seq(s, max(history_lengths)) for s in history_seqs]
    current_lengths = [len(t) for t in current_seqs]
    current_padded = [pad_seq(s, max(current_lengths)) for s in current_seqs]
    target_padded = [pad_seq(s, max(current_lengths)) for s in target_seqs]
    encoder_input = torch.LongTensor(history_padded).transpose(0, 1).cuda()
    decoder_input = torch.LongTensor(current_padded).transpose(0, 1).cuda()
    target_batches = torch.LongTensor(target_padded).transpose(0, 1).cuda()
    return encoder_input, history_lengths, decoder_input, current_lengths, target_batches


# batch_trajectory_generator
def batch_generator(batch_size, pairs, shuffle=True):
    train_queue = deque()
    initial_queue = deque()
    if shuffle:
        np.random.shuffle(pairs)
    for i, pair in enumerate(pairs):
        initial_queue.append(pair)
    while len(initial_queue) >= batch_size:
        train_seqs = []
        history_seqs = []
        current_seqs = []
        target_seqs = []
        for j in range(batch_size):
            train_pair = initial_queue.popleft()
            history_seqs.append(train_pair[0])
            current_seqs.append(train_pair[1])
            target_seqs.append(train_pair[2])
        seq_pairs = sorted(
            zip(history_seqs, current_seqs, target_seqs),
            key=lambda p: len(p[0]),
            reverse=True)
        history_seqs, current_seqs, target_seqs = zip(*seq_pairs)
        history_lengths = [len(t) for t in history_seqs]
        history_padded = [
            pad_seq(s, max(history_lengths)) for s in history_seqs
        ]
        current_lengths = [len(t) for t in current_seqs]
        current_padded = [
            pad_seq(s, max(current_lengths)) for s in current_seqs
        ]
        target_padded = [pad_seq(s, max(current_lengths)) for s in target_seqs]

        train_seqs.append(history_padded)
        train_seqs.append(history_lengths)
        train_seqs.append(current_padded)
        train_seqs.append(current_lengths)
        train_seqs.append(target_padded)

        train_queue.append(train_seqs)
    return train_queue


def generate_adj_input(loc_arr, loc_map, adj_input):
    loc_num = len(loc_arr)
    adj = np.zeros((loc_num, loc_num))
    adj[:] = np.inf
    for loc_1 in loc_arr:
        for loc_2 in loc_arr:
            try:
                if loc_2 >= loc_1:
                    adj[loc_map[loc_1]][loc_map[loc_2]] = adj_input[loc_1 - 1][
                        loc_2 - 1]
            except IndexError:
                print(loc_1, loc_2)

    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        list_user = list(user)
        while queue_left > 0:
            np.random.shuffle(list_user)
            for j, u in enumerate(list_user):
                if len(initial_queue[u]) > 0:
                    # train_queue=([(uid,initial_queue.popoleft())])
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum(
                [1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


# with reversed data
def generate_queue2(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    user_copy = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        for u in user_copy:
            if mode2 == 'train':
                initial_queue[u + len(user)] = deque(train_idx[u][1:])
        queue_left = 1
        list_user = list(user)
        tmp_user = np.arange(len(user), len(user) * 2)
        list_user.extend(tmp_user)
        while queue_left > 0:
            np.random.shuffle(list_user)
            for j, u in enumerate(list_user):
                if len(initial_queue[u]) > 0:
                    # train_queue=([(uid,initial_queue.popoleft())])
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum(
                [1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue
