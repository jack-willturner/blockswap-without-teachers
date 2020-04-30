import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
budgets=[811.4, 556.0, 404.2, 289.2,217.0,162.2]
import torchvision
import torchvision.transforms as transforms
no=5

BUDGET = int(budgets[no]*1000)

import os
import json
import argparse
from models import *
from utils  import *
from tqdm   import tqdm


model = WideResNet(40,2)

trainloader, testloader = get_cifar_loaders('~/datasets/cifar')

dataiter = iter(trainloader)
input, target = dataiter.next()
criterion = nn.CrossEntropyLoss()
for name, w in model.named_parameters():
    w.retain_grad() if 'mask' in name else None

out, _  = model(input)
loss = criterion(out,target)
loss.backward()

grads = []
for name, w in model.named_parameters():
    grads.append(w.grad.detach().view(-1)) if 'mask' in name else None

grads= torch.cat(grads).abs()
param_total = get_inf_params(model)
masks_allowed = BUDGET - (param_total-len(grads))
ranked = grads.sort()
threshold = reversed(ranked[0])[masks_allowed]
model.__prune__(threshold)

# Check there are sufficient zeros
masks= []
for name, w in model.named_parameters():
    masks.append(w.detach().view(-1)) if 'mask' in name else None

masks = torch.cat(masks)
masks.sum()

torch.save(model.state_dict(),'checkpoints/snip_mask_%d.t7' % no)




# test
model = WideResNet(40,2)

for no in range(6):
    A = torch.load('checkpoints/snip_mask_%d.t7' % no)
    model.load_state_dict(A)
    masks= []
    for name, w in model.named_parameters():
        masks.append(w.detach().view(-1)) if 'mask' in name else None

    masks = torch.cat(masks)
    print(masks.sum())














from utils import *


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
budgets=[811.4, 556.0, 404.2, 289.2,217.0,162.2]
import torchvision
import torchvision.transforms as transforms
from models import *
no=5

trainloader, testloader = get_cifar_loaders('~/datasets/cifar')
criterion=nn.CrossEntropyLoss()
model = WideResNet(40,2)

model, _ = load_model(model,'wrn_40_2_1',old_format=True)
validate(model,200,valloader=testloader,criterion=criterion)