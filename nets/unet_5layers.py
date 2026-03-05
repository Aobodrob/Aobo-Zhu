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
            # 新增一层：在原4层基础上扩展特征维度（增加一个更深层）
            in_filters = [192, 384, 768, 1024, 1024]  # 新增第5层的特征维度
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            # 新增一层：在原4层基础上扩展特征维度（增加一个更深层）
            in_filters = [192, 512, 1024, 3072, 4096]  # 新增第5层的特征维度
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # 新增一层：在原4层基础上扩展输出维度
        out_filters = [64, 128, 256, 512, 512]  # 新增第5层的输出维度

        # 上采样层：新增up_concat5，其他层保持名称不变但对应层级+1
        self.up_concat5 = unetUp(in_filters[4], out_filters[4])  # 新增最深层上采样
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])  # 原up_concat4变为第4层
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])  # 原up_concat3变为第3层
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])  # 原up_concat2变为第2层
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])  # 原up_concat1变为第1层

        # 保持与原逻辑一致（仅在resnet50且4层时使用，此处保持不变）
        if backbone == 'resnet50' and False:  # 5层时不使用额外上采样
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
        # 编码器特征提取：新增一层特征（获取前6个特征，原代码取前5个）
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5, feat6] = self.vgg.forward(inputs)[:6]  # 新增feat6
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5, feat6] = self.resnet.forward(inputs)[:6]  # 新增feat6

        # 解码器上采样：新增一层上采样（从feat6开始，原代码从feat5开始）
        up5 = self.up_concat5(feat5, feat6)  # 新增最深层上采样
        up4 = self.up_concat4(feat4, up5)  # 原up4的输入变为feat4和up5
        up3 = self.up_concat3(feat3, up4)  # 原up3的输入变为feat3和up4
        up2 = self.up_concat2(feat2, up3)  # 原up2的输入变为feat2和up3
        up1 = self.up_concat1(feat1, up2)  # 原up1的输入变为feat1和up2

        # 额外上采样层：保持与原逻辑一致（5层时不使用）
        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

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