import torch
import torch.nn as nn
import torch.nn.functional as F


class classify_net(nn.Module):
    def __init__(self):
        super(classify_net, self).__init__()
        init_channels = 1
        batch_norm = True
        self.features_extract = nn.Sequential(
            nn.Conv3d(init_channels, 32, kernel_size=1, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
            nn.Dropout3d(0.35)
        )



        self.classfication_c = nn.Sequential(
            nn.Linear(in_features=64000, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1000, out_features=2)
        )



    def forward(self, x):


        x = self.features_extract(x)
        x = x.view(x.size(0), -1)
        output = self.classfication_c(x)
        return output
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