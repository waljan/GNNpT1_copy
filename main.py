import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from statistics import mean

#own modules
from model import GCN, GCNWithJK, GraphSAGE, GraphSAGEWithJK
from Dataset_construction import DataConstructor



def train(model, train_loader, optimizer, crit):
    model.train()
    loss_all = 0
    # iterate over all batches in the training data
    for data in train_loader:
        data = data.to(device) # transfer the data to the device

        optimizer.zero_grad() # set the gradient to 0
        output = model(data) # pass the data through the model

        label = data.y.to(device) # transfer the labels to the device

        loss = crit(output, label) # conpute the loss between output and label
        loss.backward() # compute the gradient
        loss_all += data.num_graphs * loss.item()

        optimizer.step() # adjust the parameters according to the gradient

    return loss_all / len(train_loader)


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
            #             print(pred_idx)
            #             print(labels[batch][graph])
            if labels[batch][graph][pred_idx] == 1: # check if the correct label is predicted
                correct_pred += 1
            else:
                correct_pred += 0
    return correct_pred / num_graphs

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
    for k in range (4):
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



        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()  # define the loss function

        train_accs = []
        test_accs = []

        # compute training and test accuracy for every epoch
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, crit)  # train the model with the training_data
            scheduler.step()
            train_acc = evaluate(model, train_loader)  # compute the accuracy for the training data
            train_accs.append(train_acc)
            test_acc = evaluate(model,test_loader)  # compute the accuracy for the test data
            test_accs.append(test_acc)
            if epoch == num_epochs-1:
                test_res.append(test_acc)
            if epoch % 1 == 0:
                for param_group in optimizer.param_groups:
                    print('Epoch: {:03d}, lr: {:.5f}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, param_group["lr"], loss, train_acc, test_acc))
        # plot the training and test accuracies
        x = range(len(train_accs))
        ltype = [":", "-.", "--", "-"]
        plt.plot(x, train_accs, color=(1, 0, 0), linestyle=ltype[k], label="train {}".format(k))
        plt.plot(x, test_accs, color=(0, 1, 0), linestyle=ltype[k], label="test {}".format(k))
    plt.legend()
    plt.show()
    print("average val accuracy:", mean(test_res))

def train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m, folder):
    print("load data")
    raw_data = DataConstructor()
    val_res = []
    for k in range(4):
        print(k)
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder,
            k=k)  # split the data into train val and test
        print("train size:", len(train_data_list), "val size:", len(val_data_list))
        # initialize the data loaders
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
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


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()  # define the loss function

        train_accs = []
        val_accs = []

        # compute training and test accuracy for every epoch
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, crit)  # train the model with the training_data
            scheduler.step()
            train_acc = evaluate(model, train_loader)  # compute the accuracy for the training data
            train_accs.append(train_acc)
            val_acc = evaluate(model,val_loader)  # compute the accuracy for the test data
            val_accs.append(val_acc)
            if epoch == num_epochs-1:
                val_res.append(val_acc)
            if epoch % 1 == 0:
                for param_group in optimizer.param_groups:
                    print('Epoch: {:03d}, lr: {:.5f}, Loss: {:.5f}, Train Acc: {:.5f}, val Acc: {:.5f}'.format(epoch, param_group["lr"],loss, train_acc, val_acc))
        # plot the training and test accuracies
        x = range(len(train_accs))
        ltype = [":", "-.", "--","-"]
        plt.plot(x, train_accs, color = (1, 0, 0), linestyle = ltype[k], label="train {}".format(k))
        plt.plot(x, val_accs, color = (0, 1, 0), linestyle = ltype[k], label="val {}".format(k))
    plt.legend()
    plt.show()
    print("average val accuracy:", mean(val_res))







if __name__ == "__main__":
    import time
    import csv

    folder = "graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "graphs/base-dataset/"



    if folder == "graphs/paper-graphs/distance-based_10_13_14_35/":
        # 30,40,3,4,4,0.01,cpu, 0,True --> more than 85%
        # 32, 100, 4, 4, 4, 0.01, 0.95, 1, cuda, 0, True --> at epoch 60 >86%
        batch_size = 32
        num_epochs = 50
        num_layers = 4
        num_input_features = 4
        hidden = 32
        lr = 0.01
        lr_decay = 0.8
        step_size = 4 # step_size = 1, after every 1 epoch, new_lr = lr*gamma

    elif folder == "graphs/base-dataset/":
        batch_size = 32
        num_epochs = 50
        num_layers = 4
        num_input_features = 33
        hidden = 32
        lr = 0.01
        lr_decay = 0.8
        step_size = 4 # step_size = 1, after every 1 epoch, new_lr = lr*lr_decay


    device = torch.device("cuda")

    # choose one of the models by commenting out the others
    m = "GCNWithJK"
    m = "GCN"
    m = "GraphSAGE"
    m = "GraphSAGEWithJK"


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




    train_and_val(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder)


    # train_and_test(batch_size, num_epochs, num_layers, num_input_features, hidden, device, lr, step_size, lr_decay, m=m, folder=folder)


