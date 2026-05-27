import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import numpy as np


def decode(sequence, characters):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence, characters):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


def calc_acc(target, output, characters):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true, characters) == decode(pred, characters)
                  for true, pred in zip(target, output_argmax)])
    return a.mean()


def train(model, optimizer, epoch, dataloader, characters, device, scaler):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                output = model(data)
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.item()
            acc = calc_acc(target, output, characters)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


def valid(model, epoch, dataloader, characters, device):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with autocast('cuda'):
                output = model(data)
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc = calc_acc(target, output, characters)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Valid: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
