from model import GCN
from Dataset_construction import DataConstructor
import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


def evaluate(val_loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():

        for data in val_loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    correct_pred = 0
    num_graphs = 0

    for batch in range(len(labels)):
        for graph in range(len(labels[batch])):
            num_graphs += 1
            #             print(max(predictions[batch][graph]))
            pred_idx = np.argmax(predictions[batch][graph])
            #             print(pred_idx)
            #             print(labels[batch][graph])
            if labels[batch][graph][pred_idx] == 1:
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

    batch_size=30
    num_epochs = 100
    num_layers = 3
    hidden = 6
    lr = 0.008
    device = torch.device("cpu")

    print("load data")
    raw_data = DataConstructor()
    train_data_list, val_data_list, test_data_list = raw_data.get_data_list()

    train_loader = DataLoader(train_data_list, batch_size=batch_size)
    test_loader = DataLoader(test_data_list, batch_size=batch_size)
    val_loader = DataLoader(val_data_list, batch_size=batch_size)

    print("initialize model")
    model = GCN(num_layers=num_layers, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    train_accs = []
    val_accs = []
    test_accs = []

    # print(train_data_list[0].x)
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch) + "..........................")
        loss = train()
        train_acc = evaluate(train_loader)
        train_accs.append(train_acc)
        val_acc = evaluate(val_loader)
        val_accs.append(val_acc)
        test_acc = evaluate(test_loader)
        test_accs.append(test_acc)
        if epoch % 5 == 0:
            print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc))

    plot_acc(train_accs, val_accs, test_accs)