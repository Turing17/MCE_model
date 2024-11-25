import pandas as pd
import os

PATH = r'D:/Desktop/MICCAI BraTS 2021/MICCAI_FeTS2021_TrainingData/'

data = pd.read_csv(r'D:/Desktop/MICCAI BraTS 2020/MICCAI_BraTS2020_TrainingData/csv/survival_info.csv')
Br_id = data[data.columns[0]]
temp = []
for i in Br_id:
    temp.append(i.split('_')[-1])
    data.replace(i, i.split('_')[-1], inplace=True)
print(temp)
data.replace()

data.to_csv('1.csv')
# Br_id = data[data.columns[0]].tolist()
# dirs = os.listdir(PATH)
# for i in Br_id:
#     print(i)
#     for dir in dirs:
#         print(dir, '----------')
