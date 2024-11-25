import pandas as pd
from nn import data_load
from nn import net_radiomics
import torch
import torch.nn.functional as Func



path_data = "./bin/result_f/im_lab_test1.csv"
path_model_save = "./bin/model_nn/"

data = pd.read_csv(path_data)

train_txt = "./bin/feature_name/name_txt/val.txt"
file = open(train_txt, "r")
train_list = []
for i in file.read().split("\n")[:-2]:
    train_list.append(int(i))
data = data.iloc[train_list]
x, y = data_load(data)
row, _ = x.shape
print(row)
net = net_radiomics(input_num=_, hidden_num=100, output_num=3)
net.load_state_dict(torch.load(path_model_save+"ep1000-loss0.414.pth"))
net.eval()
output = net(x)
prediction = torch.max(Func.softmax(output), 1)[1]  # 1表示维度1，列，[0]表示概率值，[1]表示标签
pred_y = prediction.data.numpy()
target_y = y.data.numpy()
accuracy = sum(pred_y == target_y) / float(row)  # 预测中有多少和真实值一样
print(row)
print("acc:", accuracy)