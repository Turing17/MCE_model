import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from setting import parse_opts

sets = parse_opts()


class BraTs2020(Dataset):
    def __init__(self, dataset_path, annotation_line, train=True):
        super(BraTs2020, self).__init__()
        self.train = train
        self.dataset_path = dataset_path
        self.annotation_line = annotation_line
        self.length = len(annotation_line)
        self.P = ["flair", "t1ce", "t2", "t1"]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # BraTS20_Training_001_flair
        name = self.annotation_line[index].split()[0]
        id = 'BraTS20_Training_' + name.split('_')[2]
        npy_list = []
        p_label = 0
        for step, i in enumerate(self.P):
            if name.split('_')[-1] == i:
                p_label = step
        if 259 < int(name.split('_')[2]) < 336:
            # print(int(name.split('_')[2]))
            H_L_label = 0
        else:
            H_L_label = 1
        path_x = os.path.join(self.dataset_path, id, '{}.npy'.format(name))
        path_mask = os.path.join(self.dataset_path, id, '{}_{}.npy'.format(id, 'no_w_mask'))
        # npy = np.load(path_x).astype(np.float32)*np.load(path_mask).astype(np.float32)
        npy = np.load(path_x).astype(np.float32)
        without_w_mask = np.load(path_mask).astype(np.float32)

        # mask = np.load(path_mask).astype(np.float32)
        # npy = np.squeeze(npy)
        # npy_list.append(npy)
        # npy_x = np.stack([npy_list[0]], 0).astype(np.float32)

        npy_x = np.stack([npy], 0).astype(np.float32)
        without_w_mask = np.stack([without_w_mask], 0).astype(np.float32)

        # print(label)
        # print(npy_x.shape)
        return npy_x,without_w_mask, p_label, H_L_label


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    path_dataset = "../bin/brats_npy/brats2020_3D_160"
    path_trainval_txt = "../bin/nametxt/brats"
    with open('name_txt/val_name.txt', "r") as f:
        train_lines = f.readlines()
    train_dataset = BraTs2020(path_dataset, train_lines)
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=2,
                              pin_memory=True, drop_last=False)
    print(train_num)
    for i in train_loader:
        x, y1, y = i
        yi = y1.numpy()
        # print(x.shape)
        # print(y)
