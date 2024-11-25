import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from setting import parse_opts
import SimpleITK
sets = parse_opts()


class BraTs(Dataset):
    def __init__(self,  dataset_path, annotation_line, train=True):
        super(BraTs, self).__init__()
        self.train = train
        self.dataset_path = dataset_path
        self.annotation_line = annotation_line
        self.length = len(annotation_line)
        self.P = ["flair", "t1ce", "t2", "t1"]


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        id = self.annotation_line[index].split()[0]
        npy_list = []

        if 259 < int(id.split('_')[-1]) < 336:
            H_L_label = 0
        else:
            H_L_label = 1
        for p in self.P:
            path_x = os.path.join(self.dataset_path, id, '{}_{}.npy'.format(id, p))
            npy = np.load(path_x).astype(np.float32)
            # npy = np.squeeze(npy)
            npy_list.append(npy)
        # npy_x = np.stack([npy_list[0]], 0).astype(np.float32)

        npy_x = np.stack([npy_list[0], npy_list[1], npy_list[2]], 0).astype(np.float32)

        # print(label)
        # print(npy_x.shape)
        return npy_x, H_L_label
# if __name__ == '__main__':
#
#     from torch.utils.data import DataLoader
#     path_dataset = "../bin/brats_npy/brats2020_3D_160"
#     path_trainval_txt = "../bin/nametxt/brats"
#     with open(os.path.join(path_trainval_txt, "train_name.txt"), "r") as f:
#         train_lines = f.readlines()
#     train_dataset = BraTs2020(path_dataset, train_lines)
#     train_num = len(train_dataset)
#     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2,
#                               pin_memory=True, drop_last=False)
#     for i in train_loader:
#         x,y = i
#         print(y)
#         print()
