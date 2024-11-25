import pandas as pd
import os
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == '__main__':
    num_classes = 1
    img = cv.imread("./VGG/image_2d/flair/transverse/001.png")
    label = cv.imread("./VGG/label_2d/transverse/001.png")
    label1 = np.copy(label)
    label1[label1 == 4] = 0
    label1[label1 == 2] = 0
    label1[:, :, 0] *= 150
    label2 = np.copy(label)
    label2[label2 == 1] = 0
    label2[label2 == 2] = 0
    label2[label2 == 4] = 4
    label2[:, :, 1] *= 150
    label3 = np.copy(label)
    label3[label3 == 1] = 0
    label3[label3 == 4] = 0
    label3[label3 == 2] = 1
    label3[:, :, 0] *= 150
    label3[:, :, 1] *= 150

    label = label1 + label2+label3

    label = Image.fromarray(np.uint8(label))
    img = Image.fromarray(np.uint8(img))

    image = Image.blend(img, label, 0.5)
    image.save("D:/Desktop/mask3.png")

    plt.imshow(image)
    # plt.imshow(label2)
    plt.axis('off')
    plt.show()
