# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# ##############seq2seq_attention model#############
class Attn2(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn2, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    # out_gate=hidden state of the decoder RNN and history is
    # set of encoder outputs attn_weights size is out_state_len*
    # history_len, represent every out_state and history's weights
    # not every time input one word vector
    def forward(self, out_state, history):
        seq_len = history.size()[0]
        # state_len = out_state.size()[0]
        attn_energies = torch.zeros(seq_len).cuda()
        for i in range(seq_len):
            attn_energies[i] = self.score(out_state.squeeze(), history[i])
        return F.softmax(attn_energies, dim=0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# BLSTM-LSTM as Encoder-Decoder
class EncoderModel(nn.Module):
    def __init__(self, parameters):
        super(EncoderModel, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

        input_size = self.loc_emb_size

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(
                input_size, self.hidden_size, 1, bidirectional=True)

        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)

        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, target_len):
        # h1 = torch.zeros(1, 1, self.hidden_size)
        # c1 = torch.zeros(1, 1, self.hidden_size)
        h1 = torch.zeros(2, 1, self.hidden_size).cuda()
        c1 = torch.zeros(2, 1, self.hidden_size).cuda()
        loc_emb = self.emb_loc(loc)
        # tim_emb = self.emb_tim(tim)
        # x = torch.cat((loc_emb, tim_emb), 2)
        # x = self.dropout(loc_emb)
        x = loc_emb
        # encoder input
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len + 1], h1)
            return hidden_history, h1
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(
                x[:-target_len + 1], (h1, c1))
            return hidden_history, (h1, c1)


# ##############batched_encoder#############
class EncoderModel_batch(nn.Module):
    def __init__(self, parameters):
        super(EncoderModel_batch, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

        # input_size = self.loc_emb_size + self.tim_emb_size
        input_size = self.loc_emb_size

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(
                input_size, self.hidden_size, 1, bidirectional=True)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)

        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, history_seqs, history_lengths, batch_size):
        # h1 = torch.zeros(1, 1, self.hidden_size)
        # c1 = torch.zeros(1, 1, self.hidden_size)
        h1 = torch.zeros(2, batch_size, self.hidden_size).cuda()
        c1 = torch.zeros(2, batch_size, self.hidden_size).cuda()
        history_embedded = self.emb_loc(history_seqs)
        packed = pack_padded_sequence(history_embedded, history_lengths)
        # encoder input
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            packed_history, h1 = self.rnn_encoder(packed, h1)
            history, _ = pad_packed_sequence(packed_history)
            hidden_history = history[:, :, :
                                     self.hidden_size] + history[:, :, self.
                                                                 hidden_size:]
            return hidden_history, h1
        elif self.rnn_type == 'LSTM':
            packed_history, (h1, c1) = self.rnn_encoder(packed, (h1, c1))
            history, _ = pad_packed_sequence(packed_history)
            hidden_history = history[:, :, :
                                     self.hidden_size] + history[:, :, self.
                                                                 hidden_size:]
            return hidden_history, (h1, c1)


# ##############batched_decoder#############
class DecoderModel_batch(nn.Module):
    def __init__(self, parameters):
        super(DecoderModel_batch, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type
        self.output_size = parameters.loc_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        input_size = self.hidden_size

        if self.rnn_type == 'GRU':
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'RNN':
            self.rnn_decoder = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=1,
                dropout=parameters.dropout_p)

        self.fc_final = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, current_seq, h2, c2):
        batch_size = current_seq.size(0)
        seq_embedded = self.embedding(current_seq).view(
            1, batch_size, self.hidden_size)
        if self.rnn_type == 'GRU':
            hidden_state, h2 = self.rnn_decoder(seq_embedded, h2)
        if self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(seq_embedded, (h2, c2))
        # hidden_state = hidden_state.view(1, 1, 2, -1).squeeze(1).squeeze(0)
        # out = hidden_state[0].unsqueeze(0)
        out = hidden_state
        out = self.dropout(out)

        y = self.fc_final(out)

        return y, (h2, c2)


# ##############with_attn model#############
class DecoderModel(nn.Module):
    def __init__(self, parameters):
        super(DecoderModel, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type
        self.output_size = parameters.loc_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        input_size = self.hidden_size * 2
        self.attn = Attn2(self.attn_type, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_decoder = nn.GRU(
                input_size, self.hidden_size, 1, dropout=parameters.dropout_p)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'RNN':
            self.rnn_decoder = nn.LSTM(
                input_size, self.hidden_size, 1, dropout=parameters.dropout_p)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, last_context, h2, c2, hidden_history):
        loc_embedded = self.embedding(loc).view(1, 1, -1)
        rnn_input = torch.cat((loc_embedded, last_context.unsqueeze(0)), 2)
        if self.rnn_type == 'GRU':
            hidden_state, h2 = self.rnn_decoder(rnn_input, h2)
        if self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(rnn_input, (h2, c2))
        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1)  # no need for fc_attn
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=1)

        return score, context, (h2, c2)


# ##############without_attn model#############
class DecoderModel1(nn.Module):
    def __init__(self, parameters):
        super(DecoderModel1, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type
        self.output_size = parameters.loc_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        input_size = self.hidden_size

        if self.rnn_type == 'GRU':
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'RNN':
            self.rnn_decoder = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=1,
                dropout=parameters.dropout_p)

        self.fc_final = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, h2, c2):
        loc_embedded = self.embedding(loc).view(1, 1, -1)
        if self.rnn_type == 'GRU':
            hidden_state, h2 = self.rnn_decoder(loc_embedded, h2)
        if self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(loc_embedded, (h2, c2))
        hidden_state = hidden_state.squeeze(1)
        # hidden_state = hidden_state.view(1, 1, 2, -1).squeeze(1).squeeze(0)
        # out = hidden_state[0].unsqueeze(0)
        out = hidden_state
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=1)

        return score, (h2, c2)