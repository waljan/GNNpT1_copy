#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from main import get_opt_param

def plot_train_val(folder):
    """
    plot the training curve
    :param model: str
    :param folder: str
    :param fold: int: 0,1,2,3
    :return:
    """
    models=["GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GATNet", "GIN", "GraphNN", "NMP"]
    # models=["GIN", "GraphNN", "NMP", "GCN", "GATNet"]
    num_models = len(models)
    subplot_idx = [1,2,5,6,3,4,7,8,9,10,13,14,11,12,15,16]
    for m in models:
        subpl = 0
        for fold in range(4):
            if "paper" in folder:
                dataset = "paper/"
            elif "base" in folder:
                dataset = "base/"
            filename = m + "_test_data.csv"
            path = "out/" + dataset + m + "/" + filename
            # read test_data to get mean and sd of every data split
            with open(path, "r") as test_res:
                data = list(csv.reader(test_res))
                print("sd and mean across all test splits")
                print("sd:", data[0][0])
                print("mean:", data[0][1])
                print("sd and mean of fold " + str(fold))
                print("sd:", data[fold+1][0])
                print("mean:", data[fold+1][1])
                # test_accs = data[fold+1][2:]



            # filename = m + "_train_loss_fold" + str(fold) + ".csv"
            # path = "out/" + dataset + m + "/" + filename
            # trainloss = np.genfromtxt(path, delimiter=",")




            plt.rc("font", size=7)


            # plot train_acc, train_loss, val_acc and val_loss
            for k in ["_train_acc_fold", "_val_acc_fold", "_train_loss_fold", "_val_loss_fold"]:

                filename = m + k + str(fold) + ".csv"
                path = "out/" + dataset + m + "/" + filename
                data = np.genfromtxt(path, delimiter=",")
                plt.subplot(4,4,subplot_idx[subpl])
                subpl += 1
                x = range(len(data[0,:]))
                color = iter(plt.cm.rainbow(np.linspace(0,1,len(data[:,0]))))
                for i in range(len(data[:,0])):
                    c = next(color)
                    mx_idx = np.argmax(data[i])
                    y = data[i,:]
                    plt.plot(x, y, color=c, alpha=0.2, zorder=1)
                    # print(mx_idx, print())
                    if k=="_val_acc_fold":
                        plt.scatter(mx_idx, data[i, mx_idx], color=c, marker="o", s=6, zorder=2)
                    #     plt.scatter(mx_idx, float(test_accs[i]), color=c, marker="x", zorder=3)
                    plt.title(dataset + m + k + str(fold))
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
    models = ["GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GATNet", "NMP", "GIN", "GraphNN"]
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

    models = ["GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GATNet", "NMP", "GIN", "GraphNN"]

    outfile = "out/" + dataset + "/HyperParams.csv"
    with open(outfile, "w") as hyp:
        writer = csv.writer(hyp)
        writer.writerow(["dataset","model","fold","num_layers","hidden","lr","lr_decay","step_size","num_epochs","weight_decay"])
        for m in models:
            for fold in range(4):
                folder, m, fold, opt_hidden, opt_lr, opt_lr_decay, opt_num_epochs, opt_num_layers, opt_step_size, opt_weight_decay= get_opt_param(m, folder, fold)
                writer.writerow([folder, m, fold, opt_num_layers, opt_hidden, opt_lr, opt_lr_decay, opt_step_size, opt_num_epochs, opt_weight_decay])

def plot_mean_sd(m, folder):
    if "paper" in folder:
        dataset = "paper/"
        # path = "Hyperparameters/paper/"
        # folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    elif "base" in folder:
        dataset = "base/"
        # path = "Hyperparameters/base/"
        # folder = "pT1_dataset/graphs/base-dataset/"
    elif "CNN" in folder:
        dataset = "CNN/"

    test_res_file = "out/" +  dataset + m + "/" + m + "_test_data.csv"
    with open(test_res_file, "r") as res_file:
        reader = list(csv.reader(res_file))

        total_sd, total_avg = reader[0][0], reader[0][1]
        fold0_sd, fold0_avg, fold0_data = reader[1][0], reader[1][1], np.asarray(reader[1][2:], dtype=float)
        fold1_sd, fold1_avg, fold1_data = reader[2][0], reader[2][1], np.asarray(reader[2][2:], dtype=float)
        fold2_sd, fold2_avg, fold2_data = reader[3][0], reader[3][1], np.asarray(reader[3][2:], dtype=float)
        fold3_sd, fold3_avg, fold3_data = reader[4][0], reader[4][1], np.asarray(reader[4][2:], dtype=float)
        # print(fold3_data)
        # print(fold3_data.shape)
        data_all = [fold0_data, fold1_data, fold2_data, fold3_data]
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.violinplot(data_all, showmeans=True)
        plt.show()


            #
            #
            #
            #
            # if row[0]==m:
            #     split = ["fold 0", "fold 1", "fold 2", "fold 3", "total"]
            #     avg_acc = np.array([row[1], row[3], row[5], row[7], row[9]]).astype(np.float)
            #     sd = np.array([row[2], row[4], row[6], row[8], row[10]]).astype(np.float)
            #     plt.errorbar(split, avg_acc, sd, linestyle="None",marker="o", capsize=3)
            #     plt.ylim(0.5, 1)
            #     plt.show()











if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-d", type=str, required=True)

    # # plot all train and val accs and losses of a given model for a given dataset
    ###########################
    # parser.add_argument("--model", "-m", type=str, required=True)
    # parser.add_argument("--fold", "-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    plot_train_val(args.folder)

    # python plotResults.py -m GCN -d paper

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


    # #plot mean and sds
    # ###############################
    # parser.add_argument("--model", "-m", type=str, required=True)
    # args = parser.parse_args()
    # plot_mean_sd(args.model, args.folder)

    # python plotResults.py -m GCN -d paper