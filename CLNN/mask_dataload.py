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
    def __init__(self,  dataset_path, annotation_line, train=True):
        super(BraTs2020, self).__init__()
        self.train = train
        self.dataset_path = dataset_path
        self.annotation_line = annotation_line
        self.length = len(annotation_line)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # BraTS20_Training_001_flair
        name = self.annotation_line[index].split()[0]
        id = 'BraTS20_Training_'+name.split('_')[2]
        path_mask = os.path.join(self.dataset_path, id, '{}_{}.npy'.format(id,'seg'))
        npy = np.load(path_mask).astype(np.float32)
        npy_mask = np.stack([npy], 0).astype(np.float32)
        # print(label)
        # print(npy_x.shape)
        return npy_mask
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    path_dataset = "../bin/brats_npy/brats2020_3D_160"
    path_trainval_txt = "../bin/nametxt/brats"
    with open('name_txt/name_data.txt', "r") as f:
        train_lines = f.readlines()
    train_dataset = BraTs2020(path_dataset, train_lines)
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=2,
                              pin_memory=True, drop_last=False)
    for i in train_loader:
        x = i

        print(x.shape)


