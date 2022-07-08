import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision import models
import  torch.nn.functional as F

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg19', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            # exec("self.load_state_dict(torch.load(vgg19-dcbb9e9d.pth),)")
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
        if not requires_grad:
            for param in super.parameters():
                param.requires_grad = False
        if remove_fc:
            del self.classifier
        if show_params:
            print('Params:\n')
            for name, param in self.named_parameters():
                print(name, param.size())
            print('end\n')

    def forward(self,x):
        output = {}
        # output = []
        # params = {}
        for idx,(begin,end) in enumerate(self.ranges):
            for layer in range(begin,end):
                x = self.features[layer](x)
            output['x%d'%(idx+1)] = x
            # output.append(x)

        # for idx,(name,param) in enumerate(self.named_parameters()):
        #     params[idx] = param

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 4), (4, 9), (9, 18), (18, 27), (27, 36))
    # 'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 36))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg,batch_norm = False):
    layers = []
    in_channels = 3
    nums = 0
    for v in cfg:
        if v == 'M':
            nums = nums+1
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            if nums<4:
                conv2d = nn.Conv2d(in_channels,v,kernel_size= 3,padding=1)
                if batch_norm:
                    layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,nn.ReLU(inplace=True)]
            else :
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2,padding=2)
                if batch_norm:
                    layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)