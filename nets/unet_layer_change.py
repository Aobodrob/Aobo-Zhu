import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            # 原4层的in_filters为[192, 384, 768, 1024]，减少一层后保留前3个
            in_filters = [192, 384, 768]  # 缩减为3层对应的输入特征维度
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            # 原4层的in_filters为[192, 512, 1024, 3072]，减少一层后保留前3个
            in_filters = [192, 512, 1024]  # 缩减为3层对应的输入特征维度
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # 原out_filters为[64, 128, 256, 512]，减少一层后保留前3个
        out_filters = [64, 128, 256]  # 缩减为3层对应的输出特征维度

        # 上采样层：删除最深的up_concat4，保留3层上采样
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])  # 对应原up_concat3
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])  # 对应原up_concat2
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])  # 对应原up_concat1

        # ResNet50的额外上采样层：保持与原逻辑一致（若有需要）
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

        # 输出层：保持与原逻辑一致（通道数为类别数）
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        # 编码器特征提取：减少最深层特征，只取前4个特征（原5个，删除最深的feat5）
        if self.backbone == "vgg":
            # 原特征为[feat1, feat2, feat3, feat4, feat5]，删除feat5，保留前4个
            [feat1, feat2, feat3, feat4] = self.vgg.forward(inputs)[:4]  # 取前4个特征
        elif self.backbone == "resnet50":
            # 同上，删除最深层feat5，保留前4个特征
            [feat1, feat2, feat3, feat4] = self.resnet.forward(inputs)[:4]

        # 解码器上采样：删除原up_concat4，从feat4开始上采样（原流程从feat5开始）
        up3 = self.up_concat3(feat3, feat4)  # 原up3的输入是feat3和up4，现在直接用feat4
        up2 = self.up_concat2(feat2, up3)  # 与原逻辑一致
        up1 = self.up_concat1(feat1, up2)  # 与原逻辑一致

        # 额外上采样层（若有）：保持与原逻辑一致
        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        # 输出层：保持与原逻辑一致
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        # 保持与原逻辑一致
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        # 保持与原逻辑一致
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True