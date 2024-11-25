import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class classify_net(nn.Module):
    def __init__(self):
        super(classify_net, self).__init__()
        init_channels = 1
        batch_norm = True
        self.features_extract = nn.Sequential(
            nn.Conv3d(init_channels, 32, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(32, 64, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(64, 128, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(128, 256, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(256, 512, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
            nn.Dropout3d(0.35)
        )
        # 将卷积层的权重设置为1
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.features_extract(x)
        x = x.view(x.size(0), -1)
        return x
if __name__ == '__main__':
    path = os.path.join('../bin/brats_npy/brats2020_3D_160/BraTS20_Training_001','BraTS20_Training_001_seg.npy')

    # 创建输入张量
    input_tensor = torch.randn(2, 1, 160, 160, 160)
    # 创建3D VGG16模型实例
    model = classify_net()
    # 运行模型
    output,x = model(input_tensor)
    print(model)
    # 打印输出张量形状
    print(output.size())