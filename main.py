#!/usr/bin/python

import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from statistics import mean, stdev
import random
import os
import csv



#own modules
from model import GCN, GCNWithJK, GraphSAGE, GraphSAGEWithJK , GIN, GATNet, GraphNN, NMP
from MLP import MLP
from Save_Data_Objects import load_obj



def train(model, train_loader, optimizer, crit, device):
    """
    :param model: model that is beeing trained
    :param train_loader: data loader
    :param optimizer: which optimizer is used (Adam)
    :param crit: which loss function is used
    :param device: (str) either "cpu" or "cuda"
    """
    model.train()
    # iterate over all batches in the training data
    for data in train_loader:
        data = data.to(device) # transfer the data to the device

        optimizer.zero_grad() # set the gradient to 0
        output = model(data) # pass the data through the model

        label = data.y.to(device) # transfer the labels to the device

        loss = crit(output, torch.max(label,1)[1].long()) # compute the loss between output and label

        loss.backward() # compute the gradient

        optimizer.step() # adjust the parameters according to the gradient


def evaluate(model, val_loader, crit, device):
    """
    :param model: model that is beeing trained
    :param val_loader: data loader
    :param crit: loss function
    :param device: (str) either "cpu" or "cuda"
    :return: validation accuracy;
            validation loss;
            list that contains inforamtion about every image and how it was classified;
            number of True Pos/Neg and False Pos/neg
    """
    model.eval()

    loss_all =0
    graph_count=0
    batch_count =0
    correct_pred = 0
    img_name = [[],[], [], []]
    TP_TN_FP_FN = np.zeros((4))

    with torch.no_grad(): # gradients don't need to be calculated in evaluation

        # pass data through the model and get label and prediction
        for data in val_loader:     # iterate over every batch in validation set
            data = data.to(device)  # transfer data to device
            predT = model(data)     # pass the data through the model
            pred = predT.detach().cpu().numpy()     # store the predictions in a numpy array. For a batch of 30 graphs, the array has shape [30,2]
            labelT = data.y
            label = labelT.detach().cpu().numpy()   # store the labels of the data in a numpy array, for a batch of 30 graphs, the array has shaüe [30, 2]
            predicted_classes = (pred == pred.max(axis=1)[:,None]).astype(int)

            correct_pred += np.sum(predicted_classes[:, 0] == label[:, 0])  # count correct predictions
            # count the false negatives and false positives
            false_idx = np.argwhere(predicted_classes[:,0]!=label[:,0]).reshape(-1)
            truth = label[false_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[3] +=1
                    img_name[3].append(data.name[false_idx][c].tolist())
                if t[0]  == 0:
                    TP_TN_FP_FN[2] +=1
                    img_name[2].append(data.name[false_idx][c].tolist())
                c+=1
            # count the true positives and true negatives
            true_idx = np.argwhere(predicted_classes[:,0]==label[:,0]).reshape(-1)
            truth = label[true_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[0] +=1
                    img_name[0].append(data.name[true_idx][c].tolist())
                if t[0] == 0:
                    TP_TN_FP_FN[1] += 1
                    img_name[1].append(data.name[true_idx][c].tolist())
                c+=1
            loss = crit(predT, torch.max(labelT, 1)[1].long()) # compute the loss between output and label
            loss_all += loss.item()

            graph_count += data.num_graphs
            batch_count +=1

    acc = correct_pred / graph_count    # validation accuracy
    avg_loss = loss_all/graph_count
    return acc, avg_loss, img_name, TP_TN_FP_FN

def get_opt_param(m, folder, fold):
    """
    reads a csv file to  return the optimal parameters for a given model, dataset and data-split
    :param m: (str) indicates the model (e.g "GraphSAGE")
    :param folder: (str) indicates wheter to use 33 or 4 node features
                    4 features: "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
                    33 features: "pT1_dataset/graphs/base-dataset/"
    :param fold: (int) which fold of the 4-fold CV
    :return: folder, model, fold and hyperparameters
    """
    if folder == "pT1_dataset/graphs/base-dataset/":
        path = "Hyperparameters/base/" + m + "/"
    elif folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        path = "Hyperparameters/paper/" + m + "/"   # TODO remove old csv files or make the path more specific
    k = "fold" + str(fold)

    for f in os.listdir(path):
        if not f.startswith("."):
            if os.path.isfile(path + f) and k in f and f.endswith("undirected1.csv"): # TODO select the parameters that got the best val acc
                with open(path + f, "r") as file:
                    hp = list(csv.reader(file))
                    hp=hp[1]
                    folder = hp[0]
                    m = hp[1]
                    fold = int(hp[2])
                    opt_hidden = int(float(hp[5]))
                    opt_lr = float(hp[6])
                    opt_lr_decay = float(hp[7])
                    opt_num_epochs = int(float(hp[8]))
                    opt_num_layers = int(float(hp[9]))
                    opt_step_size = int(float(hp[10]))
                    opt_weight_decay = float(hp[11])
    return folder, m, fold, opt_hidden, opt_lr, opt_lr_decay, opt_num_epochs, opt_num_layers, opt_step_size, opt_weight_decay


def plot_acc_sep(train_accs, val_accs, test_accs):
    x = range(len(train_accs))
    plt.plot(x, train_accs, "r-.", label="train_accuracy")
    plt.plot(x, val_accs, "bx", label="val_accuracy")
    plt.plot(x, test_accs, "y", label="test_accuracy")
    plt.legend()
    plt.show()

def plot_train_test(train_accs, test_accs):
    x = range(len(train_accs))
    plt.plot(x, train_accs, "r", label="train_accuracy")
    plt.plot(x, test_accs, "y", label="test_accuracy")
    plt.legend()
    plt.show()


def train_and_test_1Fold(m, folder, fold, it, device):
    """
    train the model on the training set, select the best model weights using the validation set and test on the testset.
    :param m: (str) name of the model (e.g. "graphSAGE")
    :param folder: (str) determines the which dataset should be used
    :param fold: (int) determines the data split
    :param it: (int) indicates which run it is (multiruns)
    :param device: (str) either "cpu" or "cuda"
    :return: test accuracy, train- and val-accuracy and -loss, info about classification, number of TPTNFPFN
    """
    if folder == "pT1_dataset/graphs/base-dataset/":
        num_input_features = 33
        folder_short = "base/"
    elif folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        num_input_features = 4
        folder_short = "paper/"
    batch_size = 64

    # get the hyperparameters
    _, _, _, hidden , lr, lr_decay, num_epochs, num_layers, step_size, weight_decay = get_opt_param(m, folder, fold)
    bool=True
    while bool:
        # train and validate
        val_res, bool, train_accs, val_accs, train_losses, val_losses, img_name_res = train_and_val_1Fold(batch_size,
                                                                                                          num_epochs,
                                                                                                          num_layers,
                                                                                                          weight_decay,
                                                                                                          num_input_features,
                                                                                                          hidden,
                                                                                                          device,
                                                                                                          lr,
                                                                                                          step_size,
                                                                                                          lr_decay,
                                                                                                          m,
                                                                                                          folder,
                                                                                                          fold,
                                                                                                          opt=True, testing=True, it=it)
        if bool:
            print("FOLD" + str(fold) + "rerun due to dead neurons")
    print("val acc:", val_res[1])

    all_lists = load_obj(folder, augment=0, sd=0)
    all_test_lists = all_lists[2]
    test_data_list = all_test_lists[fold]
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

    model = torch.load("Parameters/" + folder_short + m + "_fold" + str(fold) + "it"+str(it)+".pt")
    crit = torch.nn.NLLLoss()
    test_acc, test_loss, img_name, TP_TN_FP_FN = evaluate(model, test_loader, crit, device)
    print("test_acc:", test_acc)
    print(TP_TN_FP_FN)
    return test_acc, train_accs, val_accs, train_losses, val_losses, img_name, TP_TN_FP_FN


def test(m, folder, runs, device):
    """
    write train accs and val accs of all runs to one csv file per fold
    :param model: model to train validate and test
    :param folder: dataset to use
    :param runs: number of times train, val and test is repeated
    :param device: "cuda" or "cpu"
    :return:
    """
    all_test_accs = []
    test_accs_per_fold = [[],[],[],[]]
    if folder == "pT1_dataset/graphs/base-dataset/":
        folder_short = "base/"
    elif folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        folder_short = "paper/"

    path_test = "out/" + folder_short + m + "/" + m + "_test_data.csv"

    for fold in range(4):
        print("Fold:", fold)

        path_train_acc = "out/" + folder_short + m + "/" + m + "_train_acc_fold" + str(fold) + ".csv"
        path_train_loss = "out/" + folder_short + m + "/" + m + "_train_loss_fold" + str(fold) + ".csv"

        path_val_acc = "out/" + folder_short + m + "/" + m + "_val_acc_fold" + str(fold) + ".csv"
        path_val_loss = "out/" + folder_short + m + "/" + m + "_val_loss_fold" + str(fold) + ".csv"

        path_TP_TN_FP_FN = "out/" + folder_short + m + "/" + m + "_TP_TN_FP_FN_fold" + str(fold) + ".csv"

        # write train acc, train loss, val acc and val loss to separate csv files
        with open(path_train_acc, "w") as train_acc_file, \
                open(path_train_loss, "w") as train_loss_file, \
                open(path_val_acc, "w") as val_acc_file, \
                open(path_val_loss, "w") as val_loss_file, \
                open(path_TP_TN_FP_FN, "w") as cls_file:

            train_acc_writer = csv.writer(train_acc_file)
            train_loss_writer = csv.writer(train_loss_file)

            val_acc_writer = csv.writer(val_acc_file)
            val_loss_writer = csv.writer(val_loss_file)

            cls_writer = csv.writer(cls_file)

            for it in range(runs):
                test_acc, train_accs, val_accs, train_losses, val_losses, img_name, TP_TN_FP_FN = train_and_test_1Fold(m, folder, fold=fold, device=device, it=it)

                all_test_accs.append(test_acc)
                test_accs_per_fold[fold].append(test_acc)

                train_acc_writer.writerow([i for i in train_accs])
                train_loss_writer.writerow([i for i in train_losses])

                val_acc_writer.writerow([i for i in val_accs])
                val_loss_writer.writerow([i for i in val_losses])

                final_TPTNFPFN = []
                groups = ["TP", "TN", "FP", "FN"]
                for group_idx in range(4):
                    for img in img_name[group_idx]:
                        img_str = str(img[0]) + "_" + str(img[1]) + "_" + str(img[2]) + "_" + groups[group_idx]
                        final_TPTNFPFN.append(img_str)

                cls_writer.writerow(final_TPTNFPFN)

                if it == runs-1:
                    avg = mean(test_accs_per_fold[fold])
                    sd = stdev(test_accs_per_fold[fold])
                    test_accs_per_fold[fold].append(avg)
                    test_accs_per_fold[fold].append(sd)
                    test_accs_per_fold[fold].reverse()
    # write test results (mean and sd) of every fold and total to a csv file
    with open(path_test, "w") as test_file:
        test_writer = csv.writer(test_file)
        test_writer.writerow([stdev(all_test_accs), mean(all_test_accs)])
        for fold in range(4):
            test_writer.writerow(test_accs_per_fold[fold])

    print("Results on Testset:")
    print("mean:", "\t",  mean(all_test_accs)*100)
    print("standard deviation:", "\t", stdev(all_test_accs)*100)


def train_and_val_1Fold(batch_size, num_epochs, num_layers, weight_decay, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, fold, augment=False, opt=False, testing=False, it=None):
    """
    the data of the pt1 dataset is split into train val and test in 4 different ways
    this function trains and validates using the train and val split of one of these 4 possible splits
    :param batch_size: (int)
    :param num_epochs: (int) number of epochs
    :param num_layers: (int) number of graph convolutional layers
    :param num_input_features: (int) number of node features
    :param hidden: (int) number of hidden representations per node
    :param device: (str) either "cpu" or "cuda"
    :param lr: learning rate
    :param step_size: indicates after how many epochs the learning rate is decreased by a factor of lr_decay
    :param lr_decay: factor by which the learning rate is decreased
    :param m: str, the model that should be trained
    :param folder: which dataset to use (33 or 4 node features)
    :param augment: boolean, determines whether the dataset should be augmented or not
    :param fold: int, determines which of the 4 possible splits is considered
    :param opt: (bool), determine whether the function is called during the hyperparameter optimization or not
    :return:
    """
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        folder_short = "paper/"
    if folder == "pT1_dataset/graphs/base-dataset/":
        folder_short = "base/"

    all_lists = load_obj(folder, augment=0, sd=0)
    all_train_lists = all_lists[0]
    all_val_lists = all_lists[1]
    all_test_lists = all_lists[2]
    all_train_aug_lists = all_lists[3] # contains train and augmented train graphs

    val_res = [] # will contain the best validation accuracy

    train_accs = [] # will contain the training accuracy of every epoch
    val_accs = [] # will contain the validation accuracy of every epoch

    losses = [] # will contain the training loss of every epoch
    val_losses = [] # will contain the validation loss of every epoch

    # get the training and validation data lists
    # augment data by adding/subtracting small random values from node features
    if augment:
        num_train = len(all_train_lists[fold])
        # num_train_aug = len(all_train_aug_lists[k])
        indices = list(range(0, num_train)) # get all original graphs

        # randomly select augmented graphs
        n_aug=5            # n_aug determines by which factor the dataset should be augmented
        choice = random.sample(range(1,n_aug), n_aug-1)
        for j in choice:
            indices.extend(random.sample(range(num_train*j, num_train*(j+1)),num_train))

        # create the train_data_list and val_data_list used for the DataLoader
        train_data_list = [all_train_aug_lists[fold][i] for i in indices] # contains all original graphs plus num_aug augmented graphs
        val_data_list = all_val_lists[fold]

        print("augm. train size: " + str(len(train_data_list)) + "   val size: "+ str(len(val_data_list)))


    else:
        # create the train_data_list and val_data_list used for the DataLoader
        train_data_list = all_train_lists[fold]
        val_data_list = all_val_lists[fold]
        test_data_list = all_test_lists[fold]
        # print("train size: " + str(len(train_data_list)) + "   val size: " + str(len(val_data_list)))

    # if testing:
    #     for entry in val_data_list:
    #         train_data_list.append(entry)       # add val to train data
    #     val_data_list = test_data_list          # use test_data_list for measuring the performance

    print("train size: " + str(len(train_data_list)) + "   val size: " + str(len(val_data_list)))


    # initialize train loader
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, drop_last=True)
    # initialize val loader
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)

    # initialize the model
    if m == "GCN":
        model = GCN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)  # initialize the model
    elif m == "GCNWithJK":
        model = GCNWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)  # initialize the model
    elif m == "GraphSAGE":
        model = GraphSAGE(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "GraphSAGEWithJK":
        model = GraphSAGEWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)
    elif m == "GIN":
        model = GIN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "GATNet":
        model = GATNet(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "GraphNN":
        model = GraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "NMP":
        model = NMP(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden , nn=MLP).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # define the optimizer, weight_decay corresponds to L2 regularization
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay) # learning rate decay

    crit = torch.nn.NLLLoss()

    # compute training and validation accuracy for every epoch
    for epoch in range(num_epochs):

        if epoch == 0:
            train_acc, loss, _,  _ = evaluate(model, train_loader, crit,
                                             device)  # compute the accuracy for the training data
            train_accs.append(train_acc)
            losses.append(loss)

            val_acc, val_loss, img_name, TP_TN_FP_FN  = evaluate(model, val_loader, crit,
                                                              device)  # compute the accuracy for the test data
            running_val_acc = np.asarray([0, 0, val_acc])
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            TP_TN_FP_FN_res = np.copy(TP_TN_FP_FN)
            val_res = np.copy(running_val_acc)
            img_name_res = img_name
            if testing:
                torch.save(model, "Parameters/" + folder_short + m + "_fold" + str(fold) + "it"+str(it)+".pt")
        # train the model
        train(model, train_loader, optimizer, crit, device)
        scheduler.step()
        # ge train acc and loss
        train_acc , loss, _, _ = evaluate(model, train_loader, crit, device)  # compute the accuracy for the training data
        train_accs.append(train_acc)
        losses.append(loss)

        # get validation acc and loss
        val_acc, val_loss, img_name, TP_TN_FP_FN = evaluate(model, val_loader, crit, device)  # compute the accuracy for the validation data
        running_val_acc[0] = running_val_acc[1]
        running_val_acc[1] = running_val_acc[2]
        running_val_acc[2] = val_acc


        if np.mean(running_val_acc) > np.mean(val_res) and not testing:         # if this is current best save the list of predictions and corresponding labels
            img_name_res = img_name
            TP_TN_FP_FN_res = np.copy(TP_TN_FP_FN)
            val_res = np.copy(running_val_acc)

        if running_val_acc[2] > val_res[2] and testing:  # if this is current best save the list of predictions and corresponding labels
            img_name_res = img_name
            TP_TN_FP_FN_res = np.copy(TP_TN_FP_FN)
            val_res = np.copy(running_val_acc)
            torch.save(model, "Parameters/" + folder_short + m + "_fold" + str(fold) + "it"+str(it)+".pt")

        val_accs.append(val_acc)
        val_losses.append(val_loss)

    if stdev(losses[-20:]) < 0.05 and mean(train_accs[-20:])<0.55:
        boolean = True
    else:
        boolean = False

    ####################################################################
    ###################################################################

    # plot the training and test accuracies

    ####################################################################
    if not opt:
        plt.rc("font", size=5)
        x = range(num_epochs+1)
        ltype = ["--","-"]

        plt.subplot(2, 1, 1)

        plt.plot(x, train_accs, color = (1, 0, 0), linestyle = ltype[0], label="train {}".format(fold))
        plt.plot(x, val_accs, color = (0, 1, 0), linestyle = ltype[1], label="val {}".format(fold))
        plt.ylim(0.5, 1)
        # plt.plot(x, np.mean(np.asarray(train_accs), axis=0), color=(0,0,0), label="train avg")
        # plt.plot(x, np.mean(np.asarray(val_accs), axis=0), color=(0,0,1), label="val avg")
        plt.vlines(val_accs.index(val_res[1]), 0.5, 1)

        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.legend()
        if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
            title = "paper-graphs, " + m + "   Validation accuracy: " + str(round(100*val_res[1], 2)) + "%" + "   Fold:" + str(fold)
            plt.title(title)
        if folder == "pT1_dataset/graphs/base-dataset/":
            title = "base-dataset, " + m + "   Validation accuracy: " + str(round(100*val_res[1], 2)) + "%" + "   Fold:" + str(fold)
            plt.title(title)

        #
        plot_loss = plt.subplot(2, 1, 2)

        plot_loss.plot(x, losses, color = (1, 0, 0), linestyle = ltype[0], label="train {}".format(fold))
        plot_loss.plot(x, val_losses, color = (0,1,0), linestyle = ltype[1], label="val {}".format(fold))
        # plt.plot(x, np.mean(np.asarray(losses), axis=0), color=(0,0,0), label="train avg")
        # plt.plot(x, np.mean(np.asarray(val_losses), axis=0), color=(0,0,1), label="val avg")
        plot_loss.set_title("train and val loss")
        plot_loss.legend()
        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.show()
    #######################################################################

    # compute number of false positives, false negatives, true positives and true negatives
    ######################################################################
        if not testing:
            print("true positives: ", TP_TN_FP_FN_res[0])
            print("true negatives: ", TP_TN_FP_FN_res[1])
            print("false positives: ", TP_TN_FP_FN_res[2])
            print("false_negatives: ", TP_TN_FP_FN_res[3])
            print("FP images:", img_name_res[2])
            print("FN images:", img_name_res[3])

    # print("best val accuracy:", val_res)
    return(val_res, boolean, np.asarray(train_accs), np.asarray(val_accs), np.asarray(losses), np.asarray(val_losses), img_name_res)   # the boolean tells that train_and_val was completed (good param combination)


if __name__ == "__main__":
    # choose dataset
    # folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    # folder = "pT1_dataset/graphs/base-dataset/"

    # choose device
    # device = torch.device("cpu")
    # device = torch.device("cuda")

    # choose one of the models by commenting out the others
    # m = "GCN"
    # m = "GCNWithJK"
    # m = "GraphSAGE"
    # m = "GraphSAGEWithJK"
    # m = "GIN"
    # m = "GraphNN"     # corresponds to 1-GNN
    # m = "GATNet"      # corresponds to GAT
    # m = "NMP"         # corresponds to enn

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device",  type=str, default="cuda")
    args = parser.parse_args()

    test(args.model, args.folder, args.runs, args.device)