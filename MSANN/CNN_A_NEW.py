from torch import nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.channelattention(x)
        result = out * self.spatialattention(x)
        return result


class Encoder_3D(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Encoder_3D, self).__init__()
        inter_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.cbam_block = cbam_block(out_channels)

        layers = [
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(inter_channels, eps=1e-4),
            nn.ReLU(True)

        ]

        if batch_norm:
            layers.insert(1, nn.GroupNorm(32, inter_channels))
            layers.insert(len(layers) - 1, nn.GroupNorm(32, out_channels))
        self.pub_1 = nn.Sequential(*layers)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.pub_1(x)
        x = self.cbam_block(x)
        x += residual
        x = self.pool(x)
        return x


class classify_net(nn.Module):
    def __init__(self):
        super(classify_net, self).__init__()
        init_channels = 3
        class_nums = 2
        batch_norm = True
        sample = False

        self.en1 = Encoder_3D(init_channels, 32, batch_norm)
        self.en2 = Encoder_3D(32, 64, batch_norm)
        self.en3 = Encoder_3D(64, 128, batch_norm)
        self.en4 = Encoder_3D(128, 256, batch_norm)
        self.en5 = Encoder_3D(256, 512, batch_norm)

        self.dropout = nn.Dropout3d(0.2)

        self.linear_1 = nn.Linear(in_features=64000, out_features=1000)
        self.linear_2 = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.en4(x)
        x = self.en5(x)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x_1 = x
        output = self.linear_2(x)
        return output, x_1


if __name__ == '__main__':
    # 创建输入张量
    input_tensor = torch.randn(2, 3, 160, 160, 160).cuda()

    model = classify_net().cuda()
    # 运行模型
    output, x = model(input_tensor)
    print(model)
    # 打印输出张量形状
    print(output.size())
