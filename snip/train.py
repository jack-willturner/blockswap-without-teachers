'''Train base models to later be pruned'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model',      default='resnet18', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/cifar', type=str)
parser.add_argument('--teacher_checkpoint', default='', type=str)
parser.add_argument('--student_checkpoint', default='', type=str)

parser.add_argument('--checkpoint', default='resnet18', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')

### training specific args
parser.add_argument('--epochs',     default=200, type=int)
parser.add_argument('--lr',         default=0.1)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
print(os.environ["CUDA_VISIBLE_DEVICES"])

from models import *
from utils  import *
from tqdm   import tqdm


def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2. * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(net, trainloader, criterion, optimizer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses  = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs  = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        outs_, ints_ = net(inputs)

        loss = criterion(outs_, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outs_.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)

def validate(model, epoch, valloader, criterion, checkpoint=None):
    global error_history

    batch_time = AverageMeter()
    data_time  = AverageMeter()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(valloader):
        input, target = input.to(device), target.to(device)
        # compute output
        output,_ = model(input)
        loss = criterion(output, target)
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

    error_history.append(top1.avg)
    if checkpoint:

        state = {
            'net': model.state_dict(),
            'masks': [w for name, w in model.named_parameters() if 'mask' in name],
            'epoch': epoch,
            'error_history': error_history,
        }
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global error_history

model = WideResNet(40,2)
model.to(device)

model.load_state_dict(torch.load('checkpoints/%s.t7' % args.checkpoint))

masks = []
for name, w in student.named_parameters():
    masks.append(w.detach().view(-1)) if 'mask' in name else None

masks = torch.cat(masks)
print(masks.sum())
time.sleep(3)

student = student.cuda()

trainloader, testloader = get_cifar_loaders(args.data_loc)
optimizer = optim.SGD([w for name, w in student.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_errors = []
val_losses   = []
val_errors   = []

error_history = []
for epoch in tqdm(range(args.epochs)):
    train(net, trainloader, criterion, optimizer)
    validate(net, epoch, testloader, criterion, checkpoint=args.checkpoint if epoch != 2 else args.checkpoint+'_init')
    scheduler.step()
