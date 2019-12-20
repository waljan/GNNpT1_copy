#!/usr/bin/python

import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from statistics import mean
import random
from scipy.stats import t
import argparse



#own modules
from model import GCN, GCNWithJK, GraphSAGE, GraphSAGEWithJK , OwnGraphNN, OwnGraphNN2, GATNet, GraphNN, NMP
from Dataset_construction import DataConstructor
from MLP import MLP
from Save_Data_Objects import load_obj



def train(model, train_loader, optimizer, crit, device):
    """

    :param model: (str) which model is trained
    :param train_loader:
    :param optimizer: which optimizer is used (Adam)
    :param crit: which loss function is used
    :param device: (str) either "cpu" or "cuda"
    :return:
    """
    model.train()
    # iterate over all batches in the training data
    for data in train_loader:
        data = data.to(device) # transfer the data to the device

        optimizer.zero_grad() # set the gradient to 0
        output = model(data) # pass the data through the model

        label = data.y.to(device) # transfer the labels to the device

        loss = crit(output, torch.max(label,1)[1].long()) # compute the loss between output and label
        # loss = crit(output, label)
        loss.backward() # compute the gradient

        optimizer.step() # adjust the parameters according to the gradient


def evaluate(model, val_loader, crit, device):
    model.eval()
    predictions = []
    labels = []
    loss_all =0
    graph_count=0
    batch_count =0
    correct_pred = 0
    img_name = [[],[], [], []]
    TP_TN_FP_FN = np.zeros((4))
    with torch.no_grad(): # gradients don't need to be calculated in evaluation

        # pass data through the model and get label and prediction
        for data in val_loader: # iterate over every batch in validation training set
            data = data.to(device) # trainsfer data to device
            predT = model(data)#.detach().cpu().numpy()   # pass the data through the model and store the predictions in a numpy array
                                                        # for a batch of 30 graphs, the array has shape [30,2]
            pred = predT.detach().cpu().numpy()
            labelT = data.y
            label = labelT.detach().cpu().numpy()   # store the labels of the data in a numpy array, for a batch of 30 graphs, the array has shaüe [30, 2]
            predicted_classes = (pred == pred.max(axis=1)[:,None]).astype(int)
            # predictions.append(predicted_classes) # append the prediction to the list of all predictions
            # labels.append(label)    # append the label to the list of all labels

            correct_pred += np.sum(predicted_classes[:, 0] == label[:, 0])
            # count the false negatives and false positives
            false_idx = np.argwhere(predicted_classes[:,0]!=label[:,0]).reshape(-1)
            truth = label[false_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[3] +=1
                    img_name[3].append(data.name[false_idx][c])
                if t[0]  == 0:
                    TP_TN_FP_FN[2] +=1
                    img_name[2].append(data.name[false_idx][c])
                c+=1

            true_idx = np.argwhere(predicted_classes[:,0]==label[:,0]).reshape(-1)
            truth = label[true_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[0] +=1
                    img_name[0].append(data.name[true_idx][c])
                if t[0] == 0:
                    TP_TN_FP_FN[1] += 1
                    img_name[1].append(data.name[true_idx][c])
                c+=1
            loss = crit(predT, torch.max(labelT, 1)[1].long()) # compute the loss between output and label
            # loss = crit(predT, labelT)
            # loss_all += data.num_graphs * loss.item()
            loss_all += loss.item()

            graph_count += data.num_graphs
            batch_count +=1

    acc = correct_pred / graph_count
    avg_loss = loss_all/graph_count
    return acc , avg_loss, img_name, TP_TN_FP_FN

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



def train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment, opt=False):
    print("load data")

    #load the data lists and split them into train, val, test and train-augmented
    all_lists = load_obj(folder, augment=10, sd=0.01)           ################################# choose which augmentation file
    all_train_lists = all_lists[0]
    all_val_lists = all_lists[1]
    all_test_lists = all_lists[2]
    all_train_aug_lists = all_lists[3] # contains train and augmented train graphs

    val_res = [] # will contain the validation accuracy obtained in the last epoch (one value for every cross validation run)

    train_accs = [[],[],[],[]] # every internal list will contain the training accuracy of every epoch of one single cross validation run
    val_accs = [[],[],[],[]] # every internal list will contain the validation accuracy of every epoch of one single cross validation run

    losses = [[],[],[],[]] # every internal list will contain the training loss of every epoch of one single cross validation run
    val_losses = [[],[],[],[]] # every internal list will contain the validation loss of every epoch of one single cross validation run

    batch_train_losses = [[],[],[],[]] # every internal list will contain the training loss of every batch of one single cross validation (CV) run
    batch_val_losses = [[],[],[],[]] # every internal list will contain the validation loss of every batch of one single cross validation (CV) run

    preds = [] # will contain the predictions of the last epoch of every CV run
    targets = [] # will contain the labels of the last epoch of every CV run


    # get the training and validation data lists
    for k in range(4):
        print("k:", k)

        # augment data by adding/subtracting small random values from node features
        if augment:
            num_train = len(all_train_lists[k])
            # num_train_aug = len(all_train_aug_lists[k])
            indices = list(range(0, num_train)) # get all original graphs

            # randomly select n_aug augmented graphs
            n_aug=5
            choice = random.sample(range(1,n_aug), n_aug-1)         # n_aug determines by which factor the dataset should be augmented
            for j in choice:
                indices.extend(random.sample(range(num_train*j, num_train*(j+1)),num_train))

            # create the train_data_list and val_data_list used for the DataLoader
            train_data_list = [all_train_aug_lists[k][i] for i in indices] # contains all original graphs plus num_aug augmented graphs
            val_data_list = all_val_lists[k]

            print("augm. train size:", len(train_data_list), "val size:", len(val_data_list))


        else:
            # create the train_data_list and val_data_list used for the DataLoader
            train_data_list = all_train_lists[k]
            val_data_list = all_val_lists[k]
            test_data_list = all_test_lists[k]
            print("train size:", len(train_data_list), "val size:", len(val_data_list))


        # initialize train loader
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, drop_last=True)
        # initialize val loader
        val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)


        # initialize the model
        print("initialize model", m)
        if m == "GCN":
            model = GCN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)  # initialize the model
        elif m == "GCNWithJK":
            model = GCNWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)  # initialize the model
        elif m == "GraphSAGE":
            model = GraphSAGE(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
        elif m == "GraphSAGEWithJK":
            model = GraphSAGEWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)
        elif m == "OwnGraphNN":
            model = OwnGraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)
        elif m == "OwnGraphNN2":
            model = OwnGraphNN2(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
        elif m == "GATNet":
            model = GATNet(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
        elif m == "GraphNN":
            model = GraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
        elif m == "NMP":
            model = NMP(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden , nn=MLP).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # define the optimizer, weight_decay corresponds to L2 regularization
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay) # learning rate decay
        crit = torch.nn.NLLLoss(reduction="sum")  # define the loss function

        bad_epoch = 0
        # compute training and validation accuracy for every epoch
        for epoch in range(num_epochs):
            # train the model
            train(model, train_loader, optimizer, crit, device)
            scheduler.step()

            train_acc , _, _, loss = evaluate(model, train_loader, crit, device)  # compute the accuracy for the training data
            train_accs[k].append(train_acc)
            losses[k].append(loss)


            val_acc, predictions, labels, val_loss = evaluate(model,val_loader, crit, device)  # compute the accuracy for the test data
            val_accs[k].append(val_acc)
            val_losses[k].append(val_loss)


            if epoch == num_epochs-1:
                val_res.append(val_acc)
                preds.append(predictions)
                targets.append(labels)
            if epoch % 1 == 0:
                if val_acc<0.6:
                    bad_epoch +=1
                for param_group in optimizer.param_groups:
                    print('Epoch: {:03d}, lr: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}'.format(epoch, param_group["lr"],loss, train_acc, val_acc))
            if bad_epoch == 5:
                val_res.append(val_acc)
                print("bad params, acc:", mean(val_res))
                return(mean(val_res), True, np.asarray(train_accs), np.asarray(val_accs), np.asarray(losses), np.asarray(val_losses))     # the boolean tells that train_and_val was stopped early (bad parameter combination)
    ####################################################################
    ###################################################################

    # plot the training and test accuracies

    ####################################################################
    if not opt:
        plt.rc("font", size=5)
        x = range(num_epochs)
        ltype = [":", "-.", "--","-"]

        plt.subplot(2, 1, 1)
        for k in range(4):
            plt.plot(x, train_accs[k], color = (1, 0, 0), linestyle = ltype[k], label="train {}".format(k))
            plt.plot(x, val_accs[k], color = (0, 1, 0), linestyle = ltype[k], label="val {}".format(k))
            plt.ylim(0.5, 1)
        plt.plot(x, np.mean(np.asarray(train_accs), axis=0), color=(0,0,0), label="train avg")
        plt.plot(x, np.mean(np.asarray(val_accs), axis=0), color=(0,0,1), label="val avg")
        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.legend()
        if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
            title = "paper-graphs, " + m + "   Validation accuracy: " + str(round(100*mean(val_res),2)) + "%"
            plt.title(title)
        if folder == "pT1_dataset/graphs/base-dataset/":
            title = "base-dataset, " + m + "   Validation accuracy: " + str(round(100*mean(val_res),2)) + "%"
            plt.title(title)

        #
        plot_loss = plt.subplot(2, 1, 2)
        for k in range(4):
            plot_loss.plot(x, losses[k], color = (1, 0, 0), linestyle = ltype[k], label="train {}".format(k))
            plot_loss.plot(x, val_losses[k], color = (0,1,0), linestyle = ltype[k], label="val {}".format(k))
        plt.plot(x, np.mean(np.asarray(losses), axis=0), color=(0,0,0), label="train avg")
        plt.plot(x, np.mean(np.asarray(val_losses), axis=0), color=(0,0,1), label="val avg")
        plot_loss.set_title("train and val loss")
        plot_loss.legend()
        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.show()
    #######################################################################

    # compute number of false positives, false negatives, true positives and true negatives
    ######################################################################
        for k in range(4):
            tps = 0
            tns = 0
            fps = 0
            fns = 0
            print("k =",k,"-------------")
            for pred_batch , target_batch in zip(preds[k], targets[k]):
                pr = np.asarray(pred_batch)
                pr = pr[:,0]

                tr = np.asarray(target_batch)
                tr = tr[:,0]

                for p, t in zip(pr, tr):
                    if p == 1 and t == 1:
                        tps += 1
                    if p == 1 and t == 0:
                        fps += 1
                    if p == 0 and t == 0:
                        tns += 1
                    if p == 0 and t == 1:
                        fns += 1

            print("true positives:", tps)
            print("true negatives:", tns)
            print("false positives:", fps)
            print("false_negatives:", fns)

    print("average val accuracy:", mean(val_res))
    return(mean(val_res), False, np.asarray(train_accs), np.asarray(val_accs), np.asarray(losses), np.asarray(val_losses))   # the boolean tells that train_and_val was completed (good param combination)



def train_and_val_1Fold(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment, fold, opt=False, testing=False):
    """
    the data of the pt1 dataset is split into train val and test in 4 different ways
    this function trains and validates using the train and val split of one of these 4 possible splits
    :param batch_size:
    :param num_epochs:
    :param num_layers:
    :param num_input_features:
    :param hidden:
    :param device:
    :param lr:
    :param step_size:
    :param lr_decay:
    :param m: str, the model that should be trained
    :param folder:
    :param augment: boolean, determines wheter the dataset should be augmented or not
    :param fold: int, determines which of the 4 possible splits is considered
    :param opt:
    :return:
    """
    #load the data lists and split them into train, val, test and train-augmented
    # all_lists = load_obj(folder, augment=10, sd=0.01)           ################################# choose which augmentation file
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

    if testing:
        for entry in val_data_list:
            train_data_list.append(entry)       # add val to train data
        val_data_list = test_data_list          # use test_data_list for measuring the performance

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
    elif m == "OwnGraphNN":
        model = OwnGraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden, mode="cat").to(device)
    elif m == "OwnGraphNN2":
        model = OwnGraphNN2(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "GATNet":
        model = GATNet(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "GraphNN":
        model = GraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)
    elif m == "NMP":
        model = NMP(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden , nn=MLP).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # define the optimizer, weight_decay corresponds to L2 regularization
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay) # learning rate decay

    crit = torch.nn.CrossEntropyLoss(reduction="sum")

    bad_epoch = 0
    # compute training and validation accuracy for every epoch
    for epoch in range(num_epochs):
        
        if epoch == 0:
            
            train_acc, loss , _,  _ = evaluate(model, train_loader, crit,
                                             device)  # compute the accuracy for the training data
            train_accs.append(train_acc)
            losses.append(loss)
            
            val_acc, val_loss, img_name, TP_TN_FP_FN  = evaluate(model, val_loader, crit,
                                                              device)  # compute the accuracy for the test data
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            TP_TN_FP_FN_res = TP_TN_FP_FN
            val_res = val_acc
            img_name_res = img_name
        # train the model
        train(model, train_loader, optimizer, crit, device)
        scheduler.step()
        
        train_acc , loss, _, _ = evaluate(model, train_loader, crit, device)  # compute the accuracy for the training data
        train_accs.append(train_acc)
        losses.append(loss)
        
        val_acc, val_loss, img_name, TP_TN_FP_FN = evaluate(model, val_loader, crit, device)  # compute the accuracy for the validation data
        # if len(val_accs) == 0:
        #     preds_res = predictions
        #     targets_res = labels
        #     val_res = val_acc
        if val_acc > max(val_accs):         # if this is current best save the list of predictions and corresponding labels
            img_name_res = img_name
            TP_TN_FP_FN_res = TP_TN_FP_FN
            val_res = val_acc

        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if opt:
            if epoch % 1 == 0:
                if val_acc<0.6:
                    bad_epoch +=1
                #for param_group in optimizer.param_groups:
                    #print('Epoch: {:03d}, lr: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}'.format(epoch, param_group["lr"],loss, train_acc, val_acc))
            if bad_epoch == 5:
                #print("bad params, best val acc:", val_res)
                return(val_res, True, np.asarray(train_accs), np.asarray(val_accs), np.asarray(losses), np.asarray(val_losses), None)     # the boolean tells that train_and_val was stopped early (bad parameter combination)


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
        plt.vlines(val_accs.index(val_res), 0.5, 1)

        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.legend()
        if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
            title = "paper-graphs, " + m + "   Validation accuracy: " + str(round(100*val_res,2)) + "%" + "   Fold:" + str(fold)
            plt.title(title)
        if folder == "pT1_dataset/graphs/base-dataset/":
            title = "base-dataset, " + m + "   Validation accuracy: " + str(round(100*val_res,2)) + "%" + "   Fold:" + str(fold)
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

        print("true positives: ", TP_TN_FP_FN_res[0])
        print("true negatives: ", TP_TN_FP_FN_res[1])
        print("false positives: ", TP_TN_FP_FN_res[2])
        print("false_negatives: ", TP_TN_FP_FN_res[3])
        print("FP images:", img_name_res[2])
        print("FN images:", img_name_res[3])

    # print("best val accuracy:", val_res)
    return(val_res, False, np.asarray(train_accs), np.asarray(val_accs), np.asarray(losses), np.asarray(val_losses), img_name_res)   # the boolean tells that train_and_val was completed (good param combination)




def plot_multiple_runs(num_runs, batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment):
    """
    plots average and confidence band of the train and validation accuracy and loss

    :param num_runs: integer; determines how often the model is trained and validated
    :param batch_size: integer
    :param num_epochs: integer
    :param num_layers: integer; determines the number of graph convolutions
    :param num_input_features: integer; either 4 or 33 number of input features
    :param hidden: integer; determines the number of features
    :param device: str; "cpu" or "cuda"
    :param lr: float; learning rate
    :param step_size: integer defining after how many epochs the learning rate is decrease
    :param lr_decay: float; the factor by which the learning rate is decreased
    :param m: string; model that is trained and validated
    :param folder: string; determines the dataset that is used
    :param augment: boolean; determines wheter to use the augmented data set or not
    :return:
    """
    train_accs_all_runs = np.empty((4*num_runs, num_epochs))
    val_accs_all_runs = np.empty((4*num_runs, num_epochs))
    train_losses_all_runs = np.empty((4*num_runs, num_epochs))
    val_losses_all_runs = np.empty((4*num_runs, num_epochs))
    for run in range(num_runs):
        _, _, train_accs, val_accs, losses, val_losses = train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment, opt=True)
        # train_accs, val_accs, losses and val_losses have shape (4, num_epochs)
        train_accs_all_runs[run*4: (run+1)*4, :] = train_accs*100
        val_accs_all_runs[run*4: (run+1)*4, :] = val_accs*100
        train_losses_all_runs[run*4: (run+1)*4, :] = losses
        val_losses_all_runs[run*4: (run+1)*4, :] = val_losses

    # compute confidence interval for train accuracies
    confidence = 0.95
    sd_train_acc = np.std(train_accs_all_runs, axis=0)
    mean_train_acc = np.mean(train_accs_all_runs, axis=0)
    CI_lb_train_acc = mean_train_acc - sd_train_acc*t.ppf((1 - confidence) / 2, num_epochs -1)
    CI_ub_train_acc = mean_train_acc + sd_train_acc*t.ppf((1 - confidence) / 2, num_epochs -1)

    # compute confidence interval for train losses
    confidence = 0.95
    sd_train_loss = np.std(train_losses_all_runs, axis=0)
    mean_train_loss = np.mean(train_losses_all_runs, axis=0)
    CI_lb_train_loss = mean_train_loss - sd_train_loss*t.ppf((1 - confidence) / 2, num_epochs -1)
    CI_ub_train_loss = mean_train_loss + sd_train_loss*t.ppf((1 - confidence) / 2, num_epochs -1)

    # compute confidence interval for validation accuracies
    sd_val_acc = np.std(val_accs_all_runs, axis=0)
    mean_val_acc = np.mean(val_accs_all_runs, axis=0)
    CI_lb_val_acc = mean_val_acc - sd_val_acc * t.ppf((1 - confidence) / 2, num_epochs - 1)
    CI_ub_val_acc = mean_val_acc + sd_val_acc * t.ppf((1 - confidence) / 2, num_epochs - 1)

    # compute confidence interval for validation losses
    sd_val_loss = np.std(val_losses_all_runs, axis=0)
    mean_val_loss = np.mean(val_losses_all_runs, axis=0)
    CI_lb_val_loss = mean_val_loss - sd_val_loss * t.ppf((1 - confidence) / 2, num_epochs - 1)
    CI_ub_val_loss = mean_val_loss + sd_val_loss * t.ppf((1 - confidence) / 2, num_epochs - 1)


    # plot train and validation accuracies and losses
    plt.rc("font", size=5)
    x = range(num_epochs)
    ltype = [":", "-.", "--", "-"]

    plt.subplot(2, 1, 1)
    plt.fill_between(range(num_epochs), CI_ub_train_acc, CI_lb_train_acc, facecolor="red", alpha=0.1)
    plt.plot(x, mean_train_acc, color="red", linestyle=ltype[3], label="train")

    plt.fill_between(range(num_epochs), CI_ub_val_acc, CI_lb_val_acc, facecolor="green", alpha=0.1)
    plt.plot(x, mean_val_acc, color="green", linestyle=ltype[3], label="val")
    plt.ylim(50, 100)
    plt.yticks(np.arange(50,101, 5))
    plt.xlim(0,num_epochs-1)
    plt.legend(loc="lower right")
    plt.gca().yaxis.grid(True)
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        title = "paper-graphs, " + m + ",   Mean val_acc: " + str(round(mean_val_acc[-1], 2)) + "%" + ",   sd of val_acc: " + str(np.round(sd_val_acc[-1], 2)) + ",  Total runs: "  + str(num_runs)
        plt.title(title)
    if folder == "pT1_dataset/graphs/base-dataset/":
        title = "base-dataset, " + m + ",   Mean val_acc: " + str(round(mean_val_acc[-1], 2)) + "%" + ",   sd of val_acc: " + str(np.round(sd_val_acc[-1], 2)) + ",  Total runs: "  + str(num_runs)
        plt.title(title)

    plt.subplot(2,1,2)
    plt.fill_between(range(num_epochs), CI_ub_train_loss, CI_lb_train_loss, facecolor="red", alpha=0.1)
    plt.plot(x, mean_train_loss, color="red", linestyle=ltype[3], label="train")

    plt.fill_between(range(num_epochs), CI_ub_val_loss, CI_lb_val_loss, facecolor="green", alpha=0.1)
    plt.plot(x, mean_val_loss, color="green", linestyle=ltype[3], label="val")
    plt.xlim(0, num_epochs-1)
    plt.legend(loc="upper right")
    plt.gca().yaxis.grid(True)
    plt.show()

if __name__ == "__main__":
    import time
    import csv

    # choose dataset
    folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    # folder = "pT1_dataset/graphs/base-dataset/"

    # choose device
    device = torch.device("cpu")
    device = torch.device("cuda")

    # choose one of the models by commenting out the others

    m = "GCN"
    # m = "GCNWithJK"
    m = "GraphSAGE"
    # m = "GraphSAGEWithJK"
    # m = "OwnGraphNN"
    # m = "OwnGraphNN2"
    # m = "GATNet"  # at the moment only for base

    m = "NMP"  # doesnt make much sense to pass one edge feature through a neural network

    # m = "GraphNN" # no suitable hyperparameters found so far




    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":

        if m=="GCN":
            # ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8660256410256411
            # Best parameters:  {'augment': 0, 'batch_size': 64.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.014797243234025006, 'lr_decay': 0.8006562056883522, 'm': 0, 'num_epochs': 35.0, 'num_input_features': 0, 'num_layers': 4.0, 'step_size': 6.0}
            # Time elapsed:  8320.881160974503
            # Parameter combinations evaluated:  200

            # Baby-sitting
            batch_size = 32
            num_epochs = 50
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.01
            lr_decay = 0.95
            step_size = 1
            augment=False

            # #fold 0
            # batch_size = 32
            # num_epochs = 35
            # num_layers = 3
            # num_input_features = 4
            # hidden = 16
            # lr = 0.01
            # lr_decay = 0.8
            # step_size = 4
            # augment=False

        if m=="GCNWithJK":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8698717948717949
            # Best parameters:  {'augment': 0, 'batch_size': 64.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.019668175615623458, 'lr_decay': 0.9307184807502304, 'm': 0, 'num_epochs': 55.0, 'num_input_features': 0, 'num_layers': 4.0, 'step_size': 8.0}
            # Time elapsed:  9676.831815004349
            # Parameter combinations evaluated:  200

            # Baby-sitting
            batch_size = 32
            num_epochs = 40
            num_layers = 2
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.8
            step_size = 4 # step_size = 1, after every 1 epoch, new_lr = lr*gamma
            augment=False

        if m=="GraphSAGE":
            # ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8666666666666667
            # Best parameters:  {'augment': 0, 'batch_size': 64.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.02137768522636307, 'lr_decay': 0.8613036839234021, 'm': 0, 'num_epochs': 60.0, 'num_input_features': 0, 'num_layers': 3.0, 'step_size': 8.0}
            # Time elapsed:  8008.898519039154
            # Parameter combinations evaluated:  200


# train size: 260   val size: 130
# Model: GraphSAGE   Dataset: pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/   runs evalutated: 10
# Parameters: {'hidden': 66, 'lr': 0.004087834869477245, 'lr_decay': 0.9281944375040574, 'num_epochs': 20, 'num_layers': 5, 'step_size': 3}
# score: 0.8125

            # # Baby-sitting
            # batch_size = 32
            # num_epochs = 35
            # num_layers = 3
            # num_input_features = 4
            # hidden = 32
            # lr = 0.01
            # lr_decay = 0.8
            # step_size = 4
            # augment=False

            # Baby-sitting
            batch_size = 32
            num_epochs = 40
            num_layers = 5
            num_input_features = 4
            hidden = 66
            lr = 0.0009
            lr_decay = 0.9812
            step_size = 6
            augment = False

#      Model: GraphSAGE   Dataset: pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/   runs evalutated: 10
# Parameters: {'hidden': 66, 'lr': 0.000956029296349167, 'lr_decay': 0.8124996943800818, 'num_epochs': 30, 'num_layers': 5, 'step_size': 6}
# score: 0.85546875

        if m=="GraphSAGEWithJK":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8660256410256411
            # Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.026923297608392217, 'lr_decay': 0.5887115050272453, 'm': 0, 'num_epochs': 45.0, 'num_input_features': 0, 'num_layers': 2.0, 'step_size': 10.0}
            # Time elapsed:  10604.129270076752
            # Parameter combinations evaluated:  200

            # Baby-sitting
            batch_size = 32
            num_epochs = 40
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.8
            step_size = 4
            augment=False

        if m == "OwnGraphNN":
            # ##### Results hyperopt 200 iterations; 10 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8746153846153846
            # Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.0071444025269403275, 'lr_decay': 0.8897865780584112, 'm': 0, 'num_epochs': 50.0, 'num_input_features': 0, 'num_layers': 3.0, 'step_size': 8.0}
            # Time elapsed:  16118.767907381058
            # Parameter combinations evaluated:  200

            # Baby-sitting

            # batch_size = 32
            # num_epochs = 40
            # num_layers = 3
            # num_input_features = 4
            # hidden = 32
            # lr = 0.005
            # lr_decay = 0.8
            # step_size = 4
            # augment = False

            # HyperOpt
            batch_size = 32
            num_epochs = 50
            num_layers = 3
            num_input_features = 4
            hidden = 16
            lr = 0.0071
            lr_decay = 0.89
            step_size = 8
            augment = False

        if m == "OwnGraphNN2":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.8634615384615385
            # Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.011699709513714454, 'lr_decay': 0.9342210923677433, 'm': 0, 'num_epochs': 35.0, 'num_input_features': 0, 'num_layers': 2.0, 'step_size': 2.0}
            # Time elapsed:  934.479779958725
            # Parameter combinations evaluated:  200

            # Baby-sitting
            batch_size = 32
            num_epochs = 40
            num_layers = 2
            num_input_features = 4
            hidden = 16
            lr = 0.001
            lr_decay = 0.8
            step_size = 2
            augment = False

        if m == "NMP":
            ##### Results hyperopt 200 iterations; 10 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.9019230769230769
            # Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 16.0, 'lr': 0.008720621073870023, 'lr_decay': 0.9382367187313544, 'm': 0, 'num_epochs': 50.0, 'num_input_features': 0, 'num_layers': 5.0, 'step_size': 2.0}
            # Time elapsed:  34481.11261844635
            # Parameter combinations evaluated:  200

            # batch_size = 32
            # num_epochs = 50
            # num_layers =3
            # num_input_features = 4
            # hidden = 32
            # lr = 0.005
            # lr_decay = 0.95
            # step_size = 2
            # augment = False

            batch_size = 32
            num_epochs = 30
            num_layers = 2
            num_input_features = 4
            hidden = 33
            lr = 0.009
            lr_decay = 0.5
            step_size = 6
            augment = False

# Model: NMP   Dataset: pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/   runs evalutated: 10
# Parameters: {'hidden': 33, 'lr': 0.009860755226144085, 'lr_decay': 0.5024977490178795, 'num_epochs': 30, 'num_layers': 2, 'step_size': 6}
# score: 0.83046875

        if m == "GATNet":
            batch_size = 32
            num_epochs = 50
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.95
            step_size = 2
            augment = False

######################################################################

    elif folder == "pT1_dataset/graphs/base-dataset/":

        if m=="GCN":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters:  0.9563406877360365
            # Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 99.0, 'lr': 0.003591084709792508, 'lr_decay': 0.9924941140087403, 'm': 0, 'num_epochs': 15.0, 'num_input_features': 0, 'num_layers': 2.0, 'step_size': 8.0}
            # Time elapsed:  6785.039863109589
            # Parameter combinations evaluated:  200

            # HyperOpt
            batch_size = 32
            num_epochs = 15
            num_layers = 2
            num_input_features = 33
            hidden = 99
            lr = 0.0036
            lr_decay = 0.99
            step_size = 8 # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GCNWithJK":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9608328364142318
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 99.0,
            #              'lr': 0.0024193024538765704, 'lr_decay': 0.5443459974128411, 'm': 0, 'num_epochs': 25.0,
            #              'num_input_features': 0, 'num_layers': 3.0, 'step_size': 8.0}
            # Time elapsed: 7471.490391492844
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 25
            num_layers = 3
            num_input_features = 33
            hidden = 99
            lr = 0.0024
            lr_decay = 0.544
            step_size = 8  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GraphSAGE":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.956350626118068
            # Best parameters: {'augment': 0, 'batch_size': 64.0, 'device': 0, 'folder': 0, 'hidden': 33.0,
            #              'lr': 0.014437187466177292, 'lr_decay': 0.8652153994753429, 'm': 0, 'num_epochs': 15.0,
            #              'num_input_features': 0, 'num_layers': 2.0, 'step_size': 6.0}
            # Time elapsed: 4649.555609464645
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 64
            num_epochs = 15
            num_layers = 2
            num_input_features = 33
            hidden = 33
            lr = 0.0144
            lr_decay = 0.865
            step_size = 6  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        # if m == "GraphSAGE":
        #     batch_size = 64
        #     num_epochs = 10
        #     num_layers = 2
        #     num_input_features = 33
        #     hidden = 136
        #     lr = 0.005
        #     lr_decay = 0.5
        #     step_size = 2  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
        #     augment = True

        if m == "GraphSAGEWithJK":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9563257801629895
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 99.0,
            #              'lr': 0.0014316728127871784, 'lr_decay': 0.9234235858502406, 'm': 0, 'num_epochs': 25.0,
            #              'num_input_features': 0, 'num_layers': 5.0, 'step_size': 4.0}
            # Time elapsed: 6238.133633613586
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 25
            num_layers = 5
            num_input_features = 33
            hidden = 99
            lr = 0.0014
            lr_decay = 0.92
            step_size = 4  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment=False

        if m == "GATNet":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9653498310475055
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 66.0,
            #              'lr': 0.001495015698574218, 'lr_decay': 0.6426887381471241, 'm': 0, 'num_epochs': 30.0,
            #              'num_input_features': 0, 'num_layers': 2.0, 'step_size': 10.0}
            # Time elapsed: 6889.107342720032
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 30
            num_layers = 2
            num_input_features = 33
            hidden = 66
            lr = 0.00149
            lr_decay = 0.642
            step_size = 10
            augment = False

        if m == "NMP":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.958924667064202
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 33.0,
            #              'lr': 0.001910597791897388, 'lr_decay': 0.7423309693386441, 'm': 0, 'num_epochs': 30.0,
            #              'num_input_features': 0, 'num_layers': 3.0, 'step_size': 4.0}
            # Time elapsed: 7692.55118727684
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 30
            num_layers = 3
            num_input_features = 33
            hidden = 33
            lr = 0.0019
            lr_decay = 0.742
            step_size = 4  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GraphNN":
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9569618366129994
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 132.0,
            #              'lr': 0.0015959695503089506, 'lr_decay': 0.8477208566259271, 'm': 0, 'num_epochs': 25.0,
            #              'num_input_features': 0, 'num_layers': 2.0, 'step_size': 4.0}
            # Time elapsed: 4706.736705303192
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 25
            num_layers = 2
            num_input_features = 33
            hidden = 132
            lr = 0.0016
            lr_decay = 0.848
            step_size = 4
            augment = False

        if m == "OwnGraphNN": # no augmentation, base dataset
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9582488570860664
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 99.0,
            #              'lr': 0.005154576247597213, 'lr_decay': 0.5320574578886035, 'm': 0, 'num_epochs': 25.0,
            #              'num_input_features': 0, 'num_layers': 2.0, 'step_size': 6.0}
            # Time elapsed: 4712.040412187576
            # Parameter combinations evaluated: 200

            # HyperOpt
            batch_size = 32
            num_epochs = 25
            num_layers = 2
            num_input_features = 33
            hidden = 99
            lr = 0.00515
            lr_decay = 0.532
            step_size = 6
            augment = False


        if m == "OwnGraphNN2": # no augmentation, base dataset
            ##### Results hyperopt 200 iterations; 3 runs of 4-fold cross validation per parameter combination
            # Score best parameters: 0.9576078314450408
            # Best parameters: {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 132.0,
            #              'lr': 0.005852895506109887, 'lr_decay': 0.6988460273182655, 'm': 0, 'num_epochs': 30.0,
            #              'num_input_features': 0, 'num_layers': 2.0, 'step_size': 4.0}
            # Time elapsed: 4900.7668533325195
            # Parameter combinations evaluated: 200


            batch_size = 32
            num_epochs = 30
            num_layers = 2
            num_input_features = 33
            hidden = 66
            lr = 0.0005
            lr_decay = 0.95
            step_size = 2
            augment = False

         ############### augment
        # if m == "OwnGraphNN":
        #     batch_size = 64
        #     num_epochs = 8
        #     num_layers = 2
        #     num_input_features = 33
        #     hidden = 132
        #     lr = 0.002
        #     lr_decay = 0.5
        #     step_size = 2
        #     augment = True

######################################################################


    # Not yet working: train_and_test(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder)
    fold=0
    v=0
    print("Model:", m)
    for fold in range(4):
        val_res, _, _, _, _, _, img_cls_res= train_and_val_1Fold(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment, fold, opt=False, testing=False)
        v += val_res

    print(v/4)


    # not working
    # train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder, augment=augment)
    # plot_multiple_runs(10, batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment)
