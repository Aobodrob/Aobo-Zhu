import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUpWithoutSkip(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUpWithoutSkip, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 仅上采样，不与编码器特征拼接
        x = self.up(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UnetWithoutSkip(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(UnetWithoutSkip, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            # 由于不拼接，输入通道数仅为上一层的输出通道数
            in_filters = [512, 256, 128, 64]  # 修改为解码器每层的输出通道数
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [512, 256, 128, 64]  # 修改为解码器每层的输出通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]  # 解码器每层的输出通道数

        # 上采样层（不使用跳跃连接）
        self.up4 = unetUpWithoutSkip(out_filters[3], out_filters[2])  # 对应原up_concat4
        self.up3 = unetUpWithoutSkip(out_filters[2], out_filters[1])  # 对应原up_concat3
        self.up2 = unetUpWithoutSkip(out_filters[1], out_filters[0])  # 对应原up_concat2
        self.up1 = unetUpWithoutSkip(out_filters[0], out_filters[0])  # 对应原up_concat1

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # 解码器流程：仅上采样，不拼接编码器特征
        x = feat5  # 从最深层特征开始
        x = self.up4(x)  # 对应原up4，但不拼接feat4
        x = self.up3(x)  # 对应原up3，但不拼接feat3
        x = self.up2(x)  # 对应原up2，但不拼接feat2
        x = self.up1(x)  # 对应原up1，但不拼接feat1

        if self.up_conv is not None:
            x = self.up_conv(x)

        final = self.final(x)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True