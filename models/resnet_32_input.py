'''ResNet in PyTorch.

source: https://github.com/kuangliu/pytorch-cifar.git

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18_32_input(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_32_input(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50_32_input(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_32_input(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152_32_input(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def get_resnet_32_input(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'resnet_32_input':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    resnet_dict = {
        'resnet18_32_input': resnet18_32_input,
        'resnet34_32_input': resnet34_32_input,
        'resnet50_32_input': resnet50_32_input,
        'resnet101_32_input': resnet101_32_input,
        'resnet152_32_input': resnet152_32_input,
    }
    model_fn = resnet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    model = model_fn(num_classes=num_classes)
    if pretrained:
        # Load pretrained weights if available
        # This is a placeholder as the pretrained weights are not implemented in this example.
        raise NotImplementedError("Pretrained weights not implemented for this model for now.")
    else: 
        model = model_fn(num_classes=num_classes)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    # Set the final layer to be learnable
    for param in model.linear.parameters():
        param.requires_grad = True
    
    return model