import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = self.sigmoid(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.cat((x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)), dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class DCN(nn.Module):
    def __init__(self, num_classes=2, spatial_attention=False, channel_attention=False):
        super(DCN, self).__init__()

        self.spatial_attention = spatial_attention
        self.channel_attention = channel_attention

        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=5, padding=1)
        self.batchnorm_1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3d2_1 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm3d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool3d_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3d_2 = nn.Conv3d(8, 8, kernel_size=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3d_3 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.maxpool3d_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3d2_2 = nn.Conv3d(8, 8, kernel_size=5, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv3d2_3 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            nn.Linear(8*18*18*18, num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        if self.channel_attention:
            self.channel_attention_layer = ChannelAttention(512)

        if self.spatial_attention:
            self.spatial_attention_layer = SpatialAttention()

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.batchnorm_1(x)
        x = self.relu1(x)
        x = self.maxpool3d_1(x)
        x = self.conv3d2_1(x)
        x = self.batchnorm_2(x)
        x = self.relu2(x)
        x = self.maxpool3d_2(x)
        x = self.conv3d_2(x)
        x = self.relu3(x)
        x = self.conv3d_3(x)
        x = self.maxpool3d_3(x)
        x = self.conv3d2_2(x)
        x = self.relu4(x)
        x = self.conv3d2_3(x)
        x = self.sigmoid(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)


        return x
#
#
# #
# # 创建输入张量
# input_tensor = torch.randn(2, 3, 160, 160, 160)
#
# # 创建3D VGG16模型实例
# model = DCN(2)
# #
# # # 运行模型
# output = model(input_tensor)
# print(model)
# # 打印输出张量形状
# print(output.size())
