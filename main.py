from model import GCN
from Dataset_construction import DataConstructor
import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

def train():
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


def evaluate(val_loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad(): # gradients dont need to be calculated in evaluation

        for data in val_loader: # iterate over every batch in validation training set
            data = data.to(device) # trainsfer data to device
            pred = model(data).detach().cpu().numpy()   # pass the data through the model and store the predictions in a numpy array
                                                        # for a batch of 30 graphs, the array has shape [30,2]

            label = data.y.detach().cpu().numpy()   # store the labels of the data in a numpy array,
                                                    # for a batch of 30 graphs, the array has shaüe [30, 2]
            predictions.append(pred) # append the prediction to the list of all predictions
            labels.append(label)    # append the label to the list of all labels

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

def plot_acc(train_accs, val_accs, test_accs):
    x = range(len(train_accs))
    plt.plot(x, train_accs, "r-.", label="train_accuracy")
    plt.plot(x, val_accs, "bx", label="val_accuracy")
    plt.plot(x, test_accs, "y", label="test_accuracy")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # 30,40,3,4,4,0.01,cpu, 0,True --> more than 85%
    # 32, 100, 4, 4, 4, 0.01, 0.95, 1, cuda, 0, True --> at epoch 60 >86%
    batch_size=32
    num_epochs = 100
    num_layers = 4
    num_input_features = 4
    hidden = 4
    lr = 0.01
    lr_decay = 0.95
    step_size = 1 # step_size = 1, after every 1 epoch, new_lr = lr*gamma
    device = torch.device("cuda")
    k=0
    only_train_and_test = True


    if not only_train_and_test:
        print("load data")
        raw_data = DataConstructor()
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(k=k) # split the data into train val and test

        only_train_and_test = True
        # initialize the data loaders
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)

        print("initialize model")
        model = GCN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(device) # initialize the model
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True) # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()   # define the loss function

        train_accs = []
        val_accs = []
        test_accs = []

        # print(train_data_list[0].x)
        for epoch in range(num_epochs):
            # print("Epoch: " + str(epoch) + "..........................")
            loss = train() # train the model with the training_data
            scheduler.step()
            train_acc = evaluate(train_loader) # compute the accuracy for the training data
            train_accs.append(train_acc)
            val_acc = evaluate(val_loader)  # compute the accuracy for the validation data
            val_accs.append(val_acc)
            test_acc = evaluate(test_loader) # compute the accuracy for the test data
            test_accs.append(test_acc)
            if epoch % 1 == 0:
                print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc))

        plot_acc(train_accs, val_accs, test_accs)

    else:
        print("load data")
        raw_data = DataConstructor()
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(
            k=k)  # split the data into train val and test
        # print(len(train_data_list))
        # print(len(val_data_list))
        for entry in val_data_list:
            # print(entry)
            train_data_list.append(entry)
        # print(len(train_data_list))

        # initialize the data loaders
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

        print("initialize model")
        model = GCN(num_layers=num_layers, num_input_features=num_input_features, hidden=hidden).to(
            device)  # initialize the model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # define the optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        crit = torch.nn.MSELoss()  # define the loss function

        train_accs = []
        test_accs = []

        # print(train_data_list[0].x)
        for epoch in range(num_epochs):
            # print("Epoch: " + str(epoch) + "..........................")
            loss = train()  # train the model with the training_data
            scheduler.step()
            train_acc = evaluate(train_loader)  # compute the accuracy for the training data
            train_accs.append(train_acc)
            test_acc = evaluate(test_loader)  # compute the accuracy for the test data
            test_accs.append(test_acc)
            if epoch % 1 == 0:
                print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch,
                                                                                                                 loss,
                                                                                                                 train_acc,
                                                                                                                 test_acc))

        plot_acc(train_accs, train_accs, test_accs)
