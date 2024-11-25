import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return attention * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze()).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        max_out = self.fc(self.max_pool(x).squeeze()).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        attention = torch.sigmoid(avg_out + max_out)
        return attention * x


class pub(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention(out_channels)

        layers = [
            nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(True)
        ]

        if batch_norm:
            layers.insert(1, nn.GroupNorm(32, inter_channels))
            layers.insert(len(layers) - 1, nn.GroupNorm(32, out_channels))

        self.pub_1 = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.conv1(x)
        out = self.pub_1(x)

        channel_attention = self.channel_att(out)
        spatial_attention = self.spatial_att(out)

        out = channel_attention * spatial_attention * out

        out += residual
        return out


# 更新unet3dEncoder和classify_net的定义来使用改进的pub模块

class unet3dEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x, self.pool(x)


class classify_net(nn.Module):
    def __init__(self):
        super(classify_net, self).__init__()
        init_channels = 3
        class_nums = 2
        batch_norm = True
        sample = False

        self.en1 = unet3dEncoder(init_channels, 32, batch_norm)
        self.en2 = unet3dEncoder(32, 64, batch_norm)
        self.en3 = unet3dEncoder(64, 128, batch_norm)
        self.en4 = unet3dEncoder(128, 256, batch_norm)
        self.en5 = unet3dEncoder(256, 512, batch_norm)

        self.dropout = nn.Dropout3d(0.3)

        self.linear_1 = nn.Linear(in_features=64000, out_features=1000)
        self.linear_2 = nn.Linear(in_features=1000, out_features=2)




    def forward(self, x):

        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, x = self.en4(x)
        x5, x = self.en5(x)
        x = x.view(x.size(0), -1)

        x= self.linear_1(x)
        x_1 = x
        output = self.linear_2(x)
        return output, x_1

# # 创建输入张量
# input_tensor = torch.randn(2, 3, 160, 160, 160)
# # 创建3D VGG16模型实例
# model = classify_net()
# # 运行模型
# output = model(input_tensor)
# print(model)
# # 打印输出张量形状
# print(output.size())