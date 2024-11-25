import torch
import torch.nn as nn
import torch.nn.functional as F



class pub(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)


    def forward(self, x):
        residual = self.conv1(x)
        return residual


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
        init_channels = 1
        batch_norm = True


        self.en1 = unet3dEncoder(init_channels, 32, batch_norm)
        self.en2 = unet3dEncoder(32, 64, batch_norm)
        self.en3 = unet3dEncoder(64, 128, batch_norm)
        self.en4 = unet3dEncoder(128, 256, batch_norm)
        self.en5 = unet3dEncoder(256, 512, batch_norm)
        # self.dropout = nn.Dropout3d(0.3)
        self.linear_1 = nn.Linear(in_features=64000, out_features=1000)
        self.linear_2 = nn.Linear(in_features=1000, out_features=4)




    def forward(self, x):
        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, x = self.en4(x)
        x5, x = self.en5(x)
        x = x.view(x.size(0), -1)
        x_1 = x
        x= self.linear_1(x)
        output = self.linear_2(x)
        return output, x_1
if __name__ == '__main__':

    # 创建输入张量
    input_tensor = torch.randn(2, 1, 160, 160, 160)
    # 创建3D VGG16模型实例
    model = classify_net()
    # 运行模型
    output,x = model(input_tensor)
    print(model)
    # 打印输出张量形状
    print(output.size())