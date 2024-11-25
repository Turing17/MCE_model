from torch import nn
class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels #if in_channels > out_channels else out_channels//2
        self.conv1=nn.Conv3d(in_channels,out_channels,1,stride=1)
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            # layers.insert(1, nn.BatchNorm3d(inter_channels))
            # layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
            layers.insert(1, nn.GroupNorm(32, inter_channels))
            layers.insert(len(layers) - 1, nn.GroupNorm(32, out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return (self.conv1(x)+self.pub(x))


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)

class classify_net(nn.Module):
    def __init__(self):
        super(classify_net, self).__init__()
        init_channels = 3
        class_nums = 3
        batch_norm = True
        sample = False

        self.en1 = unet3dEncoder(init_channels, 32, batch_norm)
        self.en2 = unet3dEncoder(32, 64, batch_norm)
        self.en3 = unet3dEncoder(64, 128, batch_norm)
        self.en4 = unet3dEncoder(128, 256, batch_norm)
        self.en5= unet3dEncoder(256, 512, batch_norm)


        #self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(0.3)
        self.linar=nn.Linear(in_features=64000,out_features=2)
    def forward(self, x):
        x1,x = self.en1(x)
        x2,x= self.en2(x)
        x3,x= self.en3(x)
        x4,x = self.en4(x)
        x5, x = self.en5(x)
        x = x.view(x.size(0),-1)           # 为了将前面多维度的tensor展平成一维
        output = self.linar(x)
        return output
# # # 创建输入张量
# input_tensor = torch.randn(2, 3, 160, 160, 160)
# #
# # # 创建3D VGG16模型实例
# model = classify_net()
# # #
# # # # 运行模型
# output = model(input_tensor)
# print(model)
# # # 打印输出张量形状
# print(output.size())