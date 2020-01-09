#!/usr/bin/python

import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# own modules
from Dataset_construction import split



def plot_imgs():
    img_path = "pT1_dataset/dataset/img0/img0_11_normal/img0_11_normal-image.jpg"
    # mask_path = "pT1_dataset/dataset/img0/img0_11_normal/img0_11_normal-gt.png"
    img = Image.open(img_path)
    # mask = Image.open(mask_path)
    img = np.asarray(img)
    # mask = np.asarray(mask)
    # mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
    # img = np.where(mask, img, mask)
    plt.imshow(img)
    plt.show()

    img_path = "pT1_dataset/dataset/img1/img1_11_abnormal/img1_11_abnormal-image.jpg"
    # mask_path = "pT1_dataset/dataset/img1/img1_11_abnormal/img1_11_abnormal-gt.png"
    img = Image.open(img_path)
    # mask = Image.open(mask_path)
    img = np.asarray(img)
    # mask = np.asarray(mask)
    # mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
    # img = np.where(mask, img, mask)
    plt.imshow(img)
    plt.show()

def image_sizes():
    path = "pT1_dataset/dataset/"
    smallest_width = 10000
    smallest_hight = 10000
    for patient in os.listdir(path):
        if not patient.endswith(".csv"):
            for img_folder in os.listdir(path + patient):
                img = Image.open(path+patient+ "/" + img_folder + "/" + img_folder + "-image.jpg")
                img = np.asarray(img)
                if img.shape[0] < smallest_hight:
                    smallest_hight = img.shape[0]
                    pic_h = img_folder
                if img.shape[1] < smallest_width:
                    smallest_width = img.shape[1]
                    pic_w = img_folder
    print(smallest_hight, pic_h)
    print(smallest_width, pic_w)

        # for img in os.listdir(path + paient)
        # if not f.startswith("."):
        #
        #     if os.path.isfile(path + f) and k in f:
        #         with open(path + f, "r") as file:

def get_images_of_fold(fold):
    dic = split(fold)
    path = "pT1_dataset/dataset/"
    trn,vl,tst = 0, 0, 0
    for patient in os.listdir(path):
        if not patient.endswith(".csv"):
            if dic[patient]=="train":
                for img_folder in os.listdir(path + patient):
                    trn+=1
            if dic[patient]=="val":
                for img_folder in os.listdir(path + patient):
                    vl+=1
            if dic[patient] == "test":
                for img_folder in os.listdir(path + patient):
                    tst+=1
    print (trn,vl, tst)


# def resize_images()

class CNN(nn.Module):
    """
    feed forward conv net
    """
    def __init__(selfself, output):
        pass





if __name__ == "__main__":
    # plot_imgs()
    # image_sizes()
    get_images_of_fold(0)