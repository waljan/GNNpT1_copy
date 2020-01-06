#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np


def plot_train_val(model, folder, fold):
    """
    plot the training curve
    :param model: str
    :param folder: str
    :param fold: int: 0,1,2,3
    :return:
    """
    if "paper" in folder:
        dataset = "paper/"
    elif "base" in folder:
        dataset = "base/"

    plt.rc("font", size=5)

    # training accuracy
    filename = model + "_train_acc_fold" + str(fold) + ".csv"
    path = "out/" + dataset + model + "/" + filename
    data = np.genfromtxt(path, delimiter=",")
    plt.subplot(2,2,1)
    x = range(len(data[0,:]))
    for i in range(len(data[:,0])):
        y = data[i,:]
        plt.plot(x, y, color=str(100), alpha=0.2)
        plt.title(dataset + model + "/fold" + str(fold) + "/train_acc")

    # training loss
    filename = model + "_train_loss_fold" + str(fold) + ".csv"
    path = "out/" + dataset + model + "/" + filename
    data = np.genfromtxt(path, delimiter=",")

    plt.subplot(2,2,3)
    x = range(len(data[0,:]))
    for i in range(len(data[:,0])):
        y = data[i,:]
        plt.plot(x, y, color=str(100), alpha=0.2)
        plt.title(dataset + model + "/fold" + str(fold) + "/train_loss")


    # validationa accuracy
    filename = model + "_val_acc_fold" + str(fold) + ".csv"
    path = "out/" + dataset + model + "/" + filename
    data = np.genfromtxt(path, delimiter=",")

    plt.subplot(2,2,2)
    x = range(len(data[0,:]))
    for i in range(len(data[:,0])):
        y = data[i,:]
        plt.plot(x, y, color=str(100), alpha=0.2)
        plt.title(dataset + model + "/fold" + str(fold) + "/val_acc")


    # validationa loss
    filename = model + "_val_loss_fold" + str(fold) + ".csv"
    path = "out/" + dataset + model + "/" + filename
    data = np.genfromtxt(path, delimiter=",")

    plt.subplot(2,2,4)
    x = range(len(data[0,:]))
    for i in range(len(data[:,0])):
        y = data[i,:]
        plt.plot(x, y, color=str(100), alpha=0.2)
        plt.title(dataset + model + "/fold" + str(fold) + "/val_loss")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--folder", "-d", type=str, required=True)
    parser.add_argument("--fold", "-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    model = "GCN"
    folder = "paper"
    fold = 0
    plot_train_val(args.model, args.folder, args.fold)

    # python plotResults.py -m GCNWithJK -d paper -k 0

