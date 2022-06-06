import torch
import torch.nn as nn
from fc.l2norm import L2Norm

class SSD300(nn.Module):
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        multibox = [4, 6, 6, 6, 4, 4]
        self.vgg = vgg(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = add_extras()
        self.num_classes = num_classes

        # get feature map
        loc_layers = []
        conf_layers = []
        feature_map_index = [21, -2]
        for k, v in enumerate(feature_map_index):
            # Prior Box 数量x4 4为x,y坐标以及长宽
            # print(feature_map_index[k]*self.num_classes)
            loc_layers += [nn.Conv2d(self.vgg[v].out_channels, multibox[k]*4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.vgg[v].out_channels, multibox[k]*self.num_classes
                                      ,kernel_size=3,padding=1)]
        for k,v in enumerate(self.extras[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, multibox[k]*4, kernel_size=3,padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, multibox[k]*self.num_classes, kernel_size=3,padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

    def forward(self, x):
        sources = []
        loc = []
        conf = []
        for k in range(23):
            x = self.vgg[k](x)
        # add 4_3 feature layer shape 38,38,512
        sources.append(self.L2Norm(x))
        # add con7 19,19,1024
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # add extra layers feature map
        ''' sources
            torch.Size([1, 512, 38, 38])
            torch.Size([1, 1024, 19, 19])
            torch.Size([1, 512, 10, 10])
            torch.Size([1, 256, 5, 5])
            torch.Size([1, 256, 3, 3])
            torch.Size([1, 256, 1, 1])'''
        for k, v in enumerate(self.extras):
            x = v(x)
            x = nn.ReLU(inplace=True)(x)
            if k % 2 != 0:
                sources.append(x)
        # conv to 6 feature map
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
        # 堆叠所有数据
        loc = torch.cat([v.view(v.size(0), -1) for v in loc], dim=1)
        conf = torch.cat([v.view(v.size(0), -1) for v in conf], dim=1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        output = (loc, conf)
        return output

# vgg
# 300, 300, 3 -> 300, 300, 64 -> 300, 300, 64 -> 150, 150, 64 -> 150, 150, 128 -> 150, 150, 128 -> 75, 75, 128 ->
# 75, 75, 256 -> 75, 75, 256 -> 75, 75, 256 -> 38, 38, 256 -> 38, 38, 512 -> 38, 38, 512 -> 38, 38, 512 -> 19, 19, 512 ->
# 19, 19, 512 -> 19, 19, 512 -> 19, 19, 512 -> 19, 19, 512 -> 19, 19, 1024 -> 19, 19, 1024
# add

def vgg(base, batch_norm=False):
    layers = []
    in_channels = 3
    for i in base:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        elif i == 'C':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(i), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = i
    maxpool5 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)# conv4_3 38,38,512 and conv7 19,19,1024 are feature maps  num22 and -1
    layers += [maxpool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=7)]
    model = nn.ModuleList(layers)
    return model

def add_extras():
    layers = []
    # layer8
    # 19,19,1024 ->19 19 256 -> 10, 10, 512 feature map
    layers += [nn.Conv2d(1024,256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256,512, kernel_size=3,stride=2, padding=1)]

    #layer 9
    # 10,10, 512 -> 10,10,128 -> 5,5,256 feature map
    layers += [nn.Conv2d(512, 128, kernel_size=1,stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # layer 10
    # 5,5,256 -> 5,5,128->3,3,256 feature map
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    # layer 11
    # 3,3,256 ->3,3,128->1,1,256
    layers += [nn.Conv2d(256,128,kernel_size=1,stride=1)]
    layers += [nn.Conv2d(128,256, kernel_size=3, stride=1)]
    model = nn.ModuleList(layers)
    return model




if __name__ == '__main__':
    x = torch.randn(3,3,300,300)
    model = SSD300(3)
    print(model)
    output = model(x)
    loc,conf = output
    # print(conf.shape)
    # for i in output:
    #     print(i.shape)
    # print(output)

