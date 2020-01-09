#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from main import get_opt_param

def plot_train_val(m, folder):
    """
    plot the training curve
    :param model: str
    :param folder: str
    :param fold: int: 0,1,2,3
    :return:
    """
    for fold in range(4):
        if "paper" in folder:
            dataset = "paper/"
        elif "base" in folder:
            dataset = "base/"
        filename = m + "_test_data.csv"
        path = "out/" + dataset + m + "/" + filename
        with open(path, "r") as test_res:
            data = list(csv.reader(test_res))
            print("sd and mean across all test splits")
            print("sd:", data[0][0])
            print("mean:", data[0][1])
            print("sd and mean of fold " + str(fold))
            print("sd:", data[fold+1][0])
            print("mean:", data[fold+1][1])

        plt.rc("font", size=5)

        # training accuracy
        filename = m + "_train_acc_fold" + str(fold) + ".csv"
        path = "out/" + dataset + m + "/" + filename
        data = np.genfromtxt(path, delimiter=",")
        plt.subplot(2,2,1)
        x = range(len(data[0,:]))
        for i in range(len(data[:,0])):
            y = data[i,:]
            plt.plot(x, y, color=str(100), alpha=0.2)
            plt.title(dataset + m + "/fold" + str(fold) + "/train_acc")

        # training loss
        filename = m + "_train_loss_fold" + str(fold) + ".csv"
        path = "out/" + dataset + m + "/" + filename
        data = np.genfromtxt(path, delimiter=",")

        plt.subplot(2,2,3)
        x = range(len(data[0,:]))
        for i in range(len(data[:,0])):
            y = data[i,:]
            plt.plot(x, y, color=str(100), alpha=0.2)
            plt.title(dataset + m + "/fold" + str(fold) + "/train_loss")


        # validationa accuracy
        filename = m + "_val_acc_fold" + str(fold) + ".csv"
        path = "out/" + dataset + m + "/" + filename
        data = np.genfromtxt(path, delimiter=",")

        plt.subplot(2,2,2)
        x = range(len(data[0,:]))
        for i in range(len(data[:,0])):
            y = data[i,:]
            plt.plot(x, y, color=str(100), alpha=0.2)
            plt.title(dataset + m + "/fold" + str(fold) + "/val_acc")


        # validationa loss
        filename = m + "_val_loss_fold" + str(fold) + ".csv"
        path = "out/" + dataset + m + "/" + filename
        data = np.genfromtxt(path, delimiter=",")

        plt.subplot(2,2,4)
        x = range(len(data[0,:]))
        for i in range(len(data[:,0])):
            y = data[i,:]
            plt.plot(x, y, color=str(100), alpha=0.2)
            plt.title(dataset + m + "/fold" + str(fold) + "/val_loss")
        plt.show()

def summarize_res(folder):
    """
    writes csv file containing all mean accs and sd of all folds and all models for the given dataset
    :param folder:
    :return:
    """

    if "paper" in folder:
        dataset = "paper/"
    elif "base" in folder:
        dataset = "base/"
    models = ["GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GATNet", "NMP", "OwnGraphNN", "OwnGraphNN2"]
    header1 =["", "fold0", "fold0", "fold1", "fold1", "fold2", "fold2", "fold3", "fold3", "total", "total"]
    header2 = ["","mean", "sd", "mean", "sd", "mean", "sd", "mean", "sd", "mean", "sd"]

    outfile = "out/" + dataset + "/Results.csv"
    with open(outfile, "w") as res:
        writer = csv.writer(res)
        writer.writerow(header1)
        writer.writerow(header2)
        for m in models:
            filename = m + "_test_data.csv"
            path = "out/" + dataset + m + "/" + filename
            with open(path, "r") as test_res:
                data = list(csv.reader(test_res))
                total_avg = data[0][1]
                total_sd = data[0][0]
                fold0_avg = data[1][1]
                fold0_sd = data[1][0]
                fold1_avg = data[2][1]
                fold1_sd = data[2][0]
                fold2_avg = data[3][1]
                fold2_sd = data[3][0]
                fold3_avg = data[4][1]
                fold3_sd = data[4][0]
            writer.writerow([m, fold0_avg, fold0_sd, fold1_avg, fold1_sd, fold2_avg, fold2_sd, fold3_avg, fold3_sd, total_avg, total_sd])

def summerize_HypParams(folder):
    if "paper" in folder:
        dataset = "paper/"
        path = "Hyperparameters/paper/"
        folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    elif "base" in folder:
        dataset = "base/"
        path = "Hyperparameters/base/"
        folder = "pT1_dataset/graphs/base-dataset/"

    models = ["GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GATNet", "NMP", "OwnGraphNN", "OwnGraphNN2"]

    outfile = "out/" + dataset + "/HyperParams.csv"
    with open(outfile, "w") as hyp:
        writer = csv.writer(hyp)
        writer.writerow(["dataset","model","fold","num_layers","hidden","lr","lr_decay","step_size","num_epochs"])
        for m in models:
            for fold in range(4):
                folder, m, fold, opt_hidden, opt_lr, opt_lr_decay, opt_num_epochs, opt_num_layers, opt_step_size = get_opt_param(m, folder, fold)
                writer.writerow([folder, m, fold, opt_num_layers, opt_hidden, opt_lr, opt_lr_decay, opt_step_size, opt_num_epochs])









if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-d", type=str, required=True)

    #plot all train and val accs and losses of a given model for a given dataset
    ##############################
    parser.add_argument("--model", "-m", type=str, required=True)
    # parser.add_argument("--fold", "-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    plot_train_val(args.model, args.folder)

    # # python plotResults.py -m GCN -d paper

    #create summary csv
    ##############################
    # args = parser.parse_args()
    # summarize_res(args.folder)

    # # python plotResults.py -d paper

    #create csv with all HypParams
    # ##############################
    # args = parser.parse_args()
    # summerize_HypParams(args.folder)

    # # python plotResults.py -d paper