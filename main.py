import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric import transforms
from torch_geometric.utils import true_negative, true_positive, false_negative, false_positive
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from statistics import mean
import random
import copy



#own modules
from model import GCN, GCNWithJK, GraphSAGE, GraphSAGEWithJK , OwnGraphNN, GraphNN
# from GraphConvolutions import OwnGConv
from Dataset_construction import DataConstructor



def train(model, train_loader, optimizer, crit):
    model.train()
    loss_all = 0
    count = 0
    # iterate over all batches in the training data
    for data in train_loader:
        data = data.to(device) # transfer the data to the device

        optimizer.zero_grad() # set the gradient to 0
        output = model(data) # pass the data through the model

        label = data.y.to(device) # transfer the labels to the device

        loss = crit(output, label) # compute the loss between output and label
        loss.backward() # compute the gradient
        loss_all += data.num_graphs * loss.item()

        optimizer.step() # adjust the parameters according to the gradient
        count += data.num_graphs
    return loss_all / count


def evaluate(model, val_loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad(): # gradients don't need to be calculated in evaluation

        # pass data through the model and get label and prediction
        for data in val_loader: # iterate over every batch in validation training set
            data = data.to(device) # trainsfer data to device
            pred = model(data).detach().cpu().numpy()   # pass the data through the model and store the predictions in a numpy array
                                                        # for a batch of 30 graphs, the array has shape [30,2]

            label = data.y.detach().cpu().numpy()   # store the labels of the data in a numpy array,
                                                    # for a batch of 30 graphs, the array has shaüe [30, 2]
            predictions.append(pred) # append the prediction to the list of all predictions
            labels.append(label)    # append the label to the list of all labels

    # compute the prediction accuracy
    correct_pred = 0
    num_graphs = 0
    # iterate over every graph
    for batch in range(len(labels)):
        for graph in range(len(labels[batch])):
            num_graphs += 1
            #             print(max(predictions[batch][graph]))
            pred_idx = np.argmax(predictions[batch][graph])
            predictions[batch][graph][pred_idx] = 1
            predictions[batch][graph][pred_idx-1] = 0
            #             print(pred_idx)
            #             print(labels[batch][graph])
            if labels[batch][graph][pred_idx] == 1: # check if the correct label is predicted
                correct_pred += 1
            else:
                correct_pred += 0
    acc = correct_pred / num_graphs
    return acc , predictions, labels

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


def cross_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder):
        # print("load data")
        raw_data = DataConstructor()
        k_val = []
        # initialize the model
        for k in range(4):
            print("k:", k)
            train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder, k=k) # split the data into train val and test
            print("train size:", len(train_data_list), "val size:", len(val_data_list))
            # initialize the data loaders
            train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)

            # initialize model
            print("initialize model", m)
            if m == "GCN":
                model = GCN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(
                    device)  # initialize the model
            elif m == "GCNWithJK":
                model = GCNWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden,
                                  mode="cat").to(device)  # initialize the model
            elif m == "GraphSAGE":
                model = GraphSAGE(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(
                    device)
            elif m == "GraphSAGEWithJK":
                model = GraphSAGEWithJK(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden,
                                        mode="cat").to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr) # define the optimizer
            scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
            crit = torch.nn.MSELoss()   # define the loss function

            train_accs = []
            val_accs = []

            for epoch in range(num_epochs):
                loss = train(model, train_loader, optimizer, crit) # train the model with the training_data
                scheduler.step()
                train_acc = evaluate(model, train_loader) # compute the accuracy for the training data
                train_accs.append(train_acc)
                val_acc = evaluate(model, val_loader)  # compute the accuracy for the validation data
                val_accs.append(val_acc)
                if epoch == num_epochs-1:
                    k_val.append(val_acc)
                for param_group in optimizer.param_groups:
                    # print("learning_rate epoch ", epoch, ":", param_group["lr"])
                    print('Epoch: {:03d}, Lr: {:.5f}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.format(epoch, param_group["lr"],  loss, train_acc, val_acc))
        print("average validation accuracy:", mean(k_val))
        return (mean(k_val))





def param_search(m, folder):
    batch_size_vec = [8,16, 32, 64]
    num_epochs_vec = [15, 30, 45, 60, 80]
    num_layers_vec = [2, 3, 4]
    num_input_features_vec = [4]
    hidden_vec = [4, 8, 16]
    lr_vec = [0.05, 0.01, 0.005, 0.001]
    lr_decay_vec = [1, 0.97, 0.94, 0.90, 0.8]
    step_size_vec = [1, 2]
    k_vec = [0, 1, 2, 3]

    batch_size_vec = [32]
    num_epochs_vec = [30]
    num_layers_vec = [1, 2, 3, 4 ,5]
    num_input_features_vec = [4]
    hidden_vec = [8, 16, 32, 64]
    lr_vec = [0.015]
    lr_decay_vec = [0.97]
    step_size_vec = [1]
    device = torch.device("cuda")
    params = []
    c=1
    for batch_size in batch_size_vec:
        for num_epochs in num_epochs_vec:
            for num_layers in num_layers_vec:
                for num_input_features in num_input_features_vec:
                    for hidden in hidden_vec:
                        for lr in lr_vec:
                            for lr_decay in lr_decay_vec:
                                for step_size in step_size_vec:

                                    print("progress:", c/(4*(len(batch_size_vec) * len(num_epochs_vec) * len(num_layers_vec) * len(num_input_features_vec) * len(hidden_vec)* len(lr_vec)* len(lr_decay_vec) * len(step_size_vec))))
                                    c+=1

                                    avg_val = cross_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device,
                                          lr, step_size, lr_decay, m, folder)
                                    params.append([m, batch_size, num_epochs, num_layers, num_input_features, hidden, lr, step_size, lr_decay, "Adam", avg_val])
    return(params)




def train_and_test(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder):
    print("load data")
    raw_data = DataConstructor()
    test_res = []

    train_accs = [[],[],[],[]]
    test_accs = [[],[],[],[]]
    losses = [[],[],[],[]]
    preds = []
    targets = []

    for k in range (4):
        print(k)
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder,
            k=k)  # split the data into train val and test

        # add val to train data
        for entry in val_data_list:
            train_data_list.append(entry)
        print("train size:", len(train_data_list), "test size:", len(test_data_list))
        # initialize the data loaders
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

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



        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()  # define the loss function


        # compute training and test accuracy for every epoch
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, crit)  # train the model with the training_data
            scheduler.step()
            losses[k].append(loss)
            train_acc, _, _ = evaluate(model, train_loader)  # compute the accuracy for the training data
            train_accs[k].append(train_acc)
            test_acc, predictions, labels = evaluate(model,test_loader)  # compute the accuracy for the test data
            test_accs[k].append(test_acc)
            if epoch == num_epochs-1:
                test_res.append(test_acc)
                preds.append(predictions)
                targets.append(labels)
            if epoch % 1 == 0:
                for param_group in optimizer.param_groups:
                    print('Epoch: {:03d}, lr: {:.5f}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, param_group["lr"], loss, train_acc, test_acc))

    # plot the training and test accuracies
    x = range(num_epochs)
    ltype = [":", "-.", "--", "-"]
    plt.subplot(2, 1, 1)
    for k in range(4):
        plt.plot(x, train_accs[k], color=(1, 0, 0), linestyle=ltype[k], label="train {}".format(k))
        plt.plot(x, test_accs[k], color=(0, 1, 0), linestyle=ltype[k], label="test {}".format(k))
        plt.ylim(0.5, 1)

    plt.legend()
    if folder == "graphs/paper-graphs/distance-based_10_13_14_35/":
        title = "paper-graphs, " + m + "   Test accuracy: " + str(round(100 * mean(test_res), 2)) + "%"
        plt.title(title)
    if folder == "graphs/base-dataset/":
        title = "base-dataset, " + m + "   Test accuracy: " + str(round(100 * mean(test_res), 2)) + "%"
        plt.title(title)

    plt.subplot(2, 1, 2)
    for k in range(4):
        plt.plot(x, losses[k], color=(1, 0, 0), linestyle=ltype[k])

    plt.show()


    print("average val accuracy:", mean(test_res))

def train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder, augment):
    print("load data")
    raw_data = DataConstructor()
    val_res = []

    train_accs = [[],[],[],[]]
    val_accs = [[],[],[],[]]
    losses = [[],[],[],[]]
    preds = []
    targets = []

    for k in range(4):
        print(k)
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder,
            k=k)  # split the data into train val and test
        print("train size:", len(train_data_list), "val size:", len(val_data_list))

        # augment data by adding/subtracting small random values from node features
        if augment == True:
            train_data_list_aug = copy.deepcopy(train_data_list)
            c=0
            for i in range(4): # how often the augmentation is done
                r_factors = np.random.rand(101, 101)  # create array of random numbers
                train_data_list_cp = copy.deepcopy(train_data_list)
                for graph in range(len(train_data_list)): # iterate over every graph
                    for nd in range(len(train_data_list[graph].x)): # iterate over every node
                        for f in range(len(train_data_list[graph].x[nd])): # iterate over every node feature
                            choice = r_factors[f,c]  # draw random number to determine whether to add or subtract
                            r_factor = r_factors[c+1,f]  # draw a random number to determine the value that will be added/subtracted
                            r_factor2 = r_factors[c,c+1]
                            c+=1
                            if c==100:
                                c=0
                            if choice >= 0.5:
                                train_data_list_cp[graph].x[nd][f] += r_factor2*r_factor # add a small random value to the feature "f" of node "nd" of graph "graph"
                            if choice < 0.5:
                                r_factor = 1-r_factor
                                train_data_list_cp[graph].x[nd][f] -= r_factor2*r_factor # subtract a small random value to the feature "f" of node "nd" of graph "graph"

                    train_data_list_aug.append(train_data_list_cp[graph])

            print("augm. train size:", len(train_data_list_aug), "val size:", len(val_data_list))

            # initialize train loader
            train_loader = DataLoader(train_data_list_aug, batch_size=batch_size, shuffle=True)

        else:
            # initialize train loader
            train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        # initialize val loader
        val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)

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
        elif m == "GraphNN":
            model = GraphNN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-3)  # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()  # define the loss function


        # compute training and test accuracy for every epoch
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, crit)  # train the model with the training_data
            scheduler.step()
            losses[k].append(loss)
            train_acc , _, _= evaluate(model, train_loader)  # compute the accuracy for the training data
            train_accs[k].append(train_acc)
            val_acc, predictions, labels = evaluate(model,val_loader)  # compute the accuracy for the test data
            val_accs[k].append(val_acc)

            if epoch == num_epochs-1:
                val_res.append(val_acc)
                preds.append(predictions)
                targets.append(labels)
            if epoch % 1 == 0:
                for param_group in optimizer.param_groups:
                    print('Epoch: {:03d}, lr: {:.5f}, Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}'.format(epoch, param_group["lr"],loss, train_acc, val_acc))

    # plot the training and test accuracies
    x = range(num_epochs)
    ltype = [":", "-.", "--","-"]

    plt.subplot(2, 1, 1)
    for k in range(4):
        plt.plot(x, train_accs[k], color = (1, 0, 0), linestyle = ltype[k], label="train {}".format(k))
        plt.plot(x, val_accs[k], color = (0, 1, 0), linestyle = ltype[k], label="val {}".format(k))
        plt.ylim(0.5, 1)

    plt.legend()
    if folder == "graphs/paper-graphs/distance-based_10_13_14_35/":
        title = "paper-graphs, " + m + "   Validation accuracy: " + str(round(100*mean(val_res),2)) + "%"
        plt.title(title)
    if folder == "graphs/base-dataset/":
        title = "base-dataset, " + m + "   Validation accuracy: " + str(round(100*mean(val_res),2)) + "%"
        plt.title(title)

    plt.subplot(2, 1, 2)
    for k in range(4):
        plt.plot(x, losses[k], color = (1, 0, 0), linestyle = ltype[k])

    plt.show()

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
    return(mean(val_res))







if __name__ == "__main__":
    import time
    import csv

    # choose dataset
    folder = "graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "graphs/base-dataset/"

    # choose device
    device = torch.device("cpu")
    device = torch.device("cuda")

    # choose one of the models by commenting out the others

    # m = "GCN"
    # m = "GCNWithJK"
    # m = "GraphSAGE"
    # m = "GraphSAGEWithJK"
    m = "OwnGraphNN"
    # m = "GraphNN"



    if folder == "graphs/paper-graphs/distance-based_10_13_14_35/":

        if m=="GCN":
            batch_size = 32
            num_epochs = 50
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.01
            lr_decay = 0.95
            step_size = 1
            augment=False

        if m=="GCNWithJK":
            batch_size = 32
            num_epochs = 40
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.8
            step_size = 4 # step_size = 1, after every 1 epoch, new_lr = lr*gamma
            augment=False

        if m=="GraphSAGE":
            batch_size = 32
            num_epochs = 40
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.8
            step_size = 4
            augment=False

        if m=="GraphSAGEWithJK":
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
            batch_size = 32
            num_epochs = 40
            num_layers = 3
            num_input_features = 4
            hidden = 32
            lr = 0.005
            lr_decay = 0.8
            step_size = 4
            augment = False

##### paper dataset:##################################################
    # 79-85% depending on model and initialization
    # large difference between runs (init)
    # large difference between epochs (zick zack)
    # val worse than train (overfitting to patient?) (up to 15%)
    # large variance between different train and val sets during cross validation
    # sometimes accuracy stays at 0.5. --> too large/small learning rate and/or bad initialization?
    # data augmentation most often decreases val accuracy. --> need different kind of augmentation. How many? parameters adjustment (batchsize)?
    # No clear difference between JK and not JK
    # what kind of models/convolutions would make sense?
    # How to properly validate the model? What to plot?
######################################################################

    elif folder == "graphs/base-dataset/":
        if m=="GCN": # 32, 15, 2, 33, 66, 0.005, 0.5, 4
            batch_size = 32
            num_epochs = 15
            num_layers = 2
            num_input_features = 33
            hidden = 66
            lr = 0.005
            lr_decay = 0.5
            step_size = 4 # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GCNWithJK":
            batch_size = 32
            num_epochs = 15
            num_layers = 2
            num_input_features = 33
            hidden = 66
            lr = 0.005
            lr_decay = 0.5
            step_size = 4  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GraphSAGE":
            batch_size = 32
            num_epochs = 15
            num_layers = 2
            num_input_features = 33
            hidden = 136
            lr = 0.005
            lr_decay = 0.5
            step_size = 4  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment = False

        if m == "GraphSAGEWithJK":
            # 32, 15, 3, 33, 66, 0.005, 0.2, 4 --> 94,87%
            # 64, 25, 3, 33, 66, 0.001, 0.9, 4 --> 93,46%
            batch_size = 32
            num_epochs = 15
            num_layers = 3
            num_input_features = 33
            hidden = 66
            lr = 0.005
            lr_decay = 0.2
            step_size = 4  # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay
            augment=False

        if m == "OwnGraphNN": # no augmentation, base dataset
            batch_size = 32
            num_epochs = 20
            num_layers = 3
            num_input_features = 33
            hidden = 132
            lr = 0.001
            lr_decay = 0.5
            step_size = 10
            augment = False

        # if m == "OwnGraphNN":
        #     batch_size = 64
        #     num_epochs = 20
        #     num_layers = 3
        #     num_input_features = 33
        #     hidden = 66
        #     lr = 0.001
        #     lr_decay = 0.5
        #     step_size = 5
        #     augment = True

        if m == "GraphNN":
            # 32, 15, 3, 33, 66, 0.005, 0.2, 4 --> 94,87%
            # 64, 25, 3, 33, 66, 0.001, 0.9, 4 --> 93,46%
            batch_size = 32
            num_epochs = 15
            num_layers = 3
            num_input_features = 33
            hidden = 66
            lr = 0.01
            lr_decay = 0.2
            step_size = 4  # step_size = 1, after eve

 ############ base dataset: ##########################################
            # 93-95% depending on model and initialization
            # large difference between runs (init)(but less than paper dataset)
            # val worse than train (overfitting to patient?)(up to 15%)
            # large variance between different train and val sets during cross validation (but less than paper dataset)
            # data augmentation most often decreases val accuracy. --> need different kind of augmentation. How many? parameters adjustment (batchsize)?
            # No clear difference between JK and not JK
            # what kind of models/convolutions would make sense?
            # How to properly validate the model? What to plot?
######################################################################

            #
    # # grid search for Hyperparameters
    # timestr = time.strftime(("%Y%m%d_%H%M%S"))
    # filename = "Results/Val/" + m + timestr + ".csv"
    # header = ["Model", "batch_size", "num_epochs", "num_layers", "num_input_features", "hidden", "lr", "step_size", "lr_decay", "optimizer", "avg_val"]
    # outputs = param_search(m, folder)
    # print("length of list:", len(outputs))
    # with open(filename, "w", newline="") as myfile:
    #     wr = csv.writer(myfile, delimiter=",", quoting = csv.QUOTE_ALL)
    #     wr.writerow(header)
    #     for output in outputs:
    #         wr.writerow(output)


    # val_=[]
    # for i in range(3):
    #     val_.append(train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder))
    #     print(val_)
    # print(mean(val_))
    # print(val_)
    # print(len(val_))


    train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder, augment=augment)
    #
    #
    # train_and_test(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder)

