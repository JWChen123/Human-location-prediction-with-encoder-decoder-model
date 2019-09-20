import torch
from model import EncoderModel_batch, DecoderModel_batch
from train import generate_input_long_history3, generate_queue, RnnParameterData
from train import generate_batch_long_history, random_batch, masked_cross_entropy
from train import batch_generator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time
import numpy as np
import random
import json
from sklearn.metrics import recall_score, f1_score

torch.manual_seed(1)

parameters = RnnParameterData()

USE_CUDA = True
reverse = False
bidirectional = True
candidate = parameters.data_neural.keys()
train_pairs = generate_batch_long_history(parameters.data_neural, 'train',
                                          candidate)
test_pairs = generate_batch_long_history(parameters.data_neural, 'test',
                                         candidate)

# history_batches, history_lengths, current_batches, current_lengths, target_batches = random_batch(
#     3, train_pairs)  # max_len X batch_size

# metrics = {
#     'train_loss': [],
#     'valid_loss': [],
#     'ppl': [],
#     'accuracy': [],
#     'accuracy_top5': []
# }
# initial model
encoder = EncoderModel_batch(parameters)
decoder = DecoderModel_batch(parameters)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

lr = parameters.lr

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(
    encoder.parameters(), lr=lr, weight_decay=parameters.L2)
decoder_optimizer = optim.Adam(
    decoder.parameters(), lr=lr, weight_decay=parameters.L2)

# criterion = nn.NLLLoss().cuda()

# scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
#     encoder_optimizer,
#     'max',
#     patience=parameters.lr_step,
#     factor=parameters.lr_decay,
#     threshold=1e-3)  # 动态学习率
# scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
#     decoder_optimizer,
#     'max',
#     patience=parameters.lr_step,
#     factor=parameters.lr_decay,
#     threshold=1e-3)  # 动态学习率

### batch_test
# encoder_outputs, (h1, c1) = encoder(history_batches, history_lengths, 3)
# h2 = h1[0].unsqueeze(0)
# c1 = (c1[0] + c1[1]).unsqueeze(0)
# decoder_input = current_batches[0]
# decoder_outputs = torch.zeros(max(current_lengths), 3,
#                               decoder.output_size).cuda()
# for t in range(max(current_lengths)):
#     y, (h2, c2) = decoder(decoder_input, h2, c1)
#     decoder_outputs[t] = y
#     decoder_input = target_batches[t]

# loss = masked_cross_entropy(
#     decoder_outputs.transpose(0, 1).contiguous(),
#     target_batches.transpose(0, 1).contiguous(), current_lengths)

### batch_test_end


def getacc(decoder_output, target):
    _, topi = decoder_output.data.topk(10)
    acc = np.zeros(3)
    index = topi.view(-1).cpu().numpy()
    if target == index[0] and target > 0:
        acc[0] += 1
    if target in index[:5] and target > 0:
        acc[1] += 1
    if target in index[:10] and target > 0:
        acc[2] += 1
    return acc


def train(history_batches, history_lengths, current_batches, current_lengths,
          target_batches, encoder, decoder, encoder_optimizer,
          decoder_optimizer, clip, batch_size):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    encoder_outputs, (h1, c1) = encoder(history_batches, history_lengths,
                                        batch_size)
    h2 = h1[0].unsqueeze(0)
    c1 = (c1[0] + c1[1]).unsqueeze(0)
    decoder_input = current_batches[0]
    decoder_outputs = torch.zeros(
        max(current_lengths), batch_size, decoder.output_size).cuda()
    for t in range(max(current_lengths)):
        y, (h2, c2) = decoder(decoder_input, h2, c1)
        decoder_outputs[t] = y
        decoder_input = target_batches[t]
    loss = masked_cross_entropy(
        decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(), current_lengths)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def run(parameters,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        batch_size,
        lr,
        mode='train'):
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
        run_queue = batch_generator(batch_size, train_pairs, 'True')
    if mode == 'test':
        encoder.train(False)
        decoder.train(False)
        run_queue = batch_generator(batch_size, test_pairs, 'False')
    loss_average = []
    queue_len = len(run_queue)
    idx = 0
    acc = np.zeros(3)

    for _ in range(queue_len):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        pair_seqs = run_queue.popleft()
        history_batches, history_lengths, current_batches, current_lengths, target_batches = pair_seqs
        if USE_CUDA:
            history_batches = torch.LongTensor(history_batches).transpose(
                0, 1).cuda()
            current_batches = torch.LongTensor(current_batches).transpose(
                0, 1).cuda()
            target_batches = torch.LongTensor(target_batches).transpose(
                0, 1).cuda()
        encoder_outputs, (h1, c1) = encoder(history_batches, history_lengths,
                                            batch_size)
        h2 = h1[0].unsqueeze(0)
        c1 = (c1[0] + c1[1]).unsqueeze(0)
        decoder_input = current_batches[0]
        decoder_outputs = torch.zeros(
            max(current_lengths), batch_size, decoder.output_size).cuda()
        for t in range(max(current_lengths)):
            y, (h2, c2) = decoder(decoder_input, h2, c1)
            decoder_outputs[t] = y
            decoder_input = target_batches[t]
            if mode == 'test':
                for m in range(batch_size):
                    if decoder_input[m] == 0:
                        continue
                    idx += 1  # number_sum
                    y_out = y.squeeze(0)
                    y_prob = F.log_softmax(y_out[m].unsqueeze(0), dim=1)
                    acc = np.add(acc,
                                 getacc(y_prob, target_batches[t][m].item()))

        loss = masked_cross_entropy(
            decoder_outputs.transpose(0, 1).contiguous(),
            target_batches.transpose(0, 1).contiguous(), current_lengths)
        loss_average.append(loss.item())
        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                           parameters.clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                           parameters.clip)
            encoder_optimizer.step()
            decoder_optimizer.step()
    average_loss = np.mean(loss_average)
    if mode == 'train':
        return encoder, decoder, average_loss
    if mode == 'test':
        acc_1 = acc[0] / idx
        acc_5 = acc[1] / idx
        acc_10 = acc[2] / idx
        return average_loss, acc_1, acc_5, acc_10


n_epochs = 20
batch_size = 10
start = time.time()

for epoch in range(1, n_epochs):

    # Run the train function
    encoder, decoder, loss = run(parameters, encoder, decoder,
                                 encoder_optimizer, decoder_optimizer,
                                 batch_size, lr, 'train')
    print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                           epoch, epoch / n_epochs * 100, loss)
    print(print_summary)
    average_loss, acc_1, acc_5, acc_10 = run(
        parameters, encoder, decoder, encoder_optimizer, decoder_optimizer,
        batch_size, lr, 'test')
    print('loss:%.3f acc@1:%.3f acc@5:%.3f acc@10:%.3f' % (average_loss, acc_1,
                                                           acc_5, acc_10))
