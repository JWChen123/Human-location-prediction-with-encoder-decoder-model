import torch
from model import EncoderModel, DecoderModel, DecoderModel1
from train import generate_input_long_history3, generate_queue, RnnParameterData
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import random
import json
from sklearn.metrics import recall_score, f1_score

parameters = RnnParameterData()

USE_CUDA = True
reverse = False
bidirectional = True
attn_state = False
mode_reverse = False
candidate = parameters.data_neural.keys()
data_train, train_idx = generate_input_long_history3(parameters.data_neural,
                                                     'train', candidate)
data_test, test_idx = generate_input_long_history3(parameters.data_neural,
                                                   'test', candidate)

metrics = {
    'train_loss': [],
    'valid_loss': [],
    'ppl': [],
    'accuracy': [],
    'accuracy_top5': []
}
# initial model
encoder = EncoderModel(parameters)
decoder1 = DecoderModel1(parameters)
# decoder1 = DecoderModel1(parameters)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder1.cuda()
    # decoder1.cuda()

lr = parameters.lr

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(
    encoder.parameters(), lr=lr, weight_decay=parameters.L2)
# decoder_optimizer = optim.Adam(
#     decoder.parameters(), lr=lr, weight_decay=parameters.L2)
decoder_optimizer1 = optim.Adam(
    decoder1.parameters(), lr=parameters.lr, weight_decay=parameters.L2)
criterion = nn.NLLLoss().cuda()

scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    encoder_optimizer,
    'max',
    patience=parameters.lr_step,
    factor=parameters.lr_decay,
    threshold=1e-3)  # 动态学习率
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
    decoder_optimizer1,
    'max',
    patience=parameters.lr_step,
    factor=parameters.lr_decay,
    threshold=1e-3)  # 动态学习率


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


def run_new(data,
            run_idx,
            mode,
            lr,
            clip,
            model1,
            model2,
            model1_optimizer,
            model2_optimizer,
            criterion,
            mode2=None):
    run_queue = None
    if mode == 'train':
        model1.train(True)
        model2.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model1.train(False)
        model2.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    users_ppl = {}
    ground_target = []
    pred_target = []
    for _ in range(queue_len):
        acc = np.zeros(3)
        ppl_list = []
        model1_optimizer.zero_grad()
        model2_optimizer.zero_grad()
        loss = 0
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0]
            users_ppl[u] = ppl_list
        # reverse mode
        if mode_reverse:
            use_reverse = random.random() < 0.5
            if use_reverse:
                loc = np.array(data[u][i]['loc'])
                loc = torch.LongTensor(loc[::-1].copy()).cuda()
            else:
                loc = data[u][i]['loc'].cuda()
        else:
            loc = data[u][i]['loc'].cuda()
        target = data[u][i]['target'].cuda()
        target_len = target.data.size()[0]
        target = target.reshape(target_len, 1)
        # encoder_outputs,hidden_state
        history, (h1, c1) = model1(loc, target_len)

        # decoder_input,context
        decoder_input = torch.LongTensor([[loc[-target_len]]])
        if attn_state is True:
            decoder_context = torch.zeros(1, model2.hidden_size)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
                decoder_context = decoder_context.cuda()
            h2 = h1
            c2 = c1
            for di in range(target_len):
                decoder_output, decoder_context, (h2, c2) = model2(
                    decoder_input, decoder_context, h2, c2, history)
                loss += criterion(decoder_output, target[di])
                if mode == 'test':
                    acc = np.add(acc, getacc(decoder_output,
                                             target[di].item()))
                    users_ppl[u].append(
                        math.log10(
                            np.exp(
                                criterion(decoder_output, target[di]).item())))
                decoder_input = target[di]
        else:
            decoder_input = decoder_input.cuda()
            # use BLSTM as encoder
            if bidirectional is True:
                h2 = h1[0].unsqueeze(0)
                c1 = (c1[0] + c1[1]).unsqueeze(0)
            for di in range(target_len):
                decoder_output, (h2, c2) = model2(decoder_input, h2, c1)
                loss += criterion(decoder_output, target[di])
                if mode == 'test':
                    acc = np.add(acc, getacc(decoder_output,
                                             target[di].item()))
                    users_ppl[u].append(
                        math.log10(
                            np.exp(
                                criterion(decoder_output, target[di]).item())))
                    ground_target.append(target[di].item())
                    pred_target.append(
                        decoder_output.data.topk(10)[1].view(-1).cpu().numpy()[
                            0])
                decoder_input = target[di]

        if mode == 'train':
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model1.parameters(), clip)
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), clip)
            model1_optimizer.step()
            model2_optimizer.step()

        if mode == 'test':
            users_acc[u][0] += target_len
            users_acc[u][1] += acc[0]
            users_acc[u][2] += acc[1]
            # users_acc[u][3] += acc[2]
        avgloss = loss.data.item() / target_len
        total_loss.append(avgloss)
    epoch_loss = np.mean(total_loss)
    if mode == 'train':
        return epoch_loss, model1, model2
    elif mode == 'test':
        users_rnn_acc = {}
        users_rnn_acc_top5 = {}
        users_rnn_ppl = {}
        # users_rnn_acc_top10 = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            top5_acc = users_acc[u][2] / users_acc[u][0]
            # top10_acc = users_acc[u][3] / users_acc[u][0]
            tmp_ppl = np.mean(users_ppl[u])
            users_rnn_acc[u] = tmp_acc
            users_rnn_acc_top5[u] = top5_acc
            users_rnn_ppl[u] = tmp_ppl
            # users_rnn_acc_top10[u] = top10_acc
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        avg_ppl = np.mean([users_rnn_ppl[x] for x in users_ppl])
        recall = recall_score(ground_target, pred_target, average='micro')
        f1 = f1_score(ground_target, pred_target, average='micro')

        avg_acc_top5 = np.mean(
            [users_rnn_acc_top5[x] for x in users_rnn_acc_top5])
        # avg_acc_top10 = np.mean(
        #     [users_rnn_acc_top10[x] for x in users_rnn_acc_top10])
        return epoch_loss, avg_acc, avg_acc_top5, avg_ppl, recall, f1  # avg_acc_top10


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


# Begin training
n_epochs = 20
start = time.time()
for epoch in range(1, n_epochs):

    # Run the train function

    loss, encoder, decoder = run_new(
        data_train, train_idx, 'train', lr, parameters.clip, encoder, decoder1,
        encoder_optimizer, decoder_optimizer1, criterion)

    # Keep track of loss
    if epoch == 0:
        continue

    print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                           epoch, epoch / n_epochs * 100, loss)
    metrics['train_loss'].append(loss)
    print(print_summary)
    valid_loss, avg_acc, avg_acc_top5, avg_ppl, recall, f1 = run_new(
        data_test, test_idx, 'test', lr, parameters.clip, encoder, decoder1,
        encoder_optimizer, decoder_optimizer1, criterion)
    print('loss:%.3f acc@1:%.3f acc@5:%.3f ppl:%.3f recall:%.3f f1:%.3f' %
          (valid_loss, avg_acc, avg_acc_top5, avg_ppl, recall, f1))
    metrics['valid_loss'].append(valid_loss)
    metrics['accuracy'].append(avg_acc)
    metrics['accuracy_top5'].append(avg_acc_top5)
    metrics['ppl'].append(avg_ppl)
    save_name_tmp1 = 'ep_' + str(epoch) + 'encoder.m'
    save_name_tmp2 = 'ep_' + str(epoch) + 'decoder.m'
    torch.save(encoder.state_dict(), './results/deepmove/' + save_name_tmp1)
    torch.save(decoder.state_dict(), './results/deepmove/' + save_name_tmp2)
    scheduler1.step(avg_acc)
    scheduler2.step(avg_acc)
    lr_last = lr
    lr = (encoder_optimizer.param_groups[0]['lr'] +
          decoder_optimizer1.param_groups[0]['lr']) / 2
    if lr_last > lr:
        load_epoch = np.argmax(metrics['accuracy'])
        load_name_tmp1 = 'ep_' + str(load_epoch + 1) + 'encoder.m'
        load_name_tmp2 = 'ep_' + str(load_epoch + 1) + 'decoder.m'
        encoder.load_state_dict(
            torch.load('./results/deepmove/' + load_name_tmp1))
        decoder.load_state_dict(
            torch.load('./results/deepmove/' + load_name_tmp2))
        print('load epoch={} model state'.format(load_epoch + 1))
# save metrics
metrics_view = {
    'train_loss': [],
    'valid_loss': [],
    'accuracy': [],
    'accuracy_top5': [],
    'ppl': []
}
for key in metrics_view:
    metrics_view[key] = metrics[key]
json.dump(
    {
        'metrics': metrics_view,
        'param': {
            'hidden_size': parameters.hidden_size,
            'L2': parameters.L2,
            'lr': lr,
            'loc_emb': parameters.loc_emb_size,
            'dropout': parameters.dropout_p,
            'clip': parameters.clip,
            'lr_step': parameters.lr_step,
            'lr_decay': parameters.lr_decay
        }
    },
    fp=open('./results/' + 'Deepmove' + '.txt', 'w'),
    indent=4)
