import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class Shortcut(nn.Module):
    def __init__(self, in_planes, planes, expansion=1, kernel_size=1, stride=1, bias=False, mode='train'):
        super(Shortcut, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask1 = nn.Parameter(torch.ones_like(self.conv1.weight))

    def conv1func(self, x):
        return F.conv2d(x, self.conv1.weight * self.mask1, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)

    def forward(self, x):
        return self.conv1func(x)

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.mask1.grad), threshold).float(), self.mask1))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode='train'):
        super(BasicBlock, self).__init__()
        self.mode  = mode
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask1 = nn.Parameter(torch.ones_like(self.conv1.weight))

        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2 = nn.Parameter(torch.ones_like(self.conv2.weight))
        self.shortcut = nn.Sequential()

        self.equalInOut = (in_planes == planes)
        self.shortcut = (not self.equalInOut) and Shortcut(in_planes, planes, self.expansion, kernel_size=1, stride=stride, bias=False) or None

    def conv1func(self, x):
        return F.conv2d(x, self.conv1.weight * self.mask1, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)

    def conv2func(self, x):
        return F.conv2d(x, self.conv2.weight * self.mask2, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding)


    def forward(self, x):

        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv2func(F.relu(self.bn2(self.conv1func(out if self.equalInOut else x))))
        return torch.add(x if self.equalInOut else self.shortcut(x), out)

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.mask1.grad), threshold).float(), self.mask1))
        self.mask2 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.mask2.grad), threshold).float(), self.mask2))

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, mode='train'):
        super(WideResNet, self).__init__()
        self.in_planes = 64
        self.mode = mode

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        self.conv1  = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.mask1 = nn.Parameter(torch.ones_like(self.conv1.weight))
        self.block1 = self._make_layer(n, nChannels[0], nChannels[1], stride=1)
        self.block2 = self._make_layer(n, nChannels[1], nChannels[2], stride=2)
        self.block3 = self._make_layer(n, nChannels[2], nChannels[3], stride=2)
        self.bn1    = nn.BatchNorm2d(nChannels[3])
        self.linear = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                truncated_normal_(m.weight.data, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def conv1func(self, x):
        return F.conv2d(x, self.conv1.weight * self.mask1, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)

    def _make_layer(self, n, in_planes, out_planes, stride):
        layers = []
        for i in range(int(n)):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1func(x)

        activations = []

        for sub_block in self.block1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block3:
            out = sub_block(out)
            activations.append(out)

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.linear(out)
        return out, activations

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.mask1.grad), threshold).float(), self.mask1))
        layers = [self.block1, self.block2, self.block3]
        for layer in layers:
            for sub_block in layer:
                sub_block.__prune__(threshold)