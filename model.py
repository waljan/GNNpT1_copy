import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, JumpingKnowledge


class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural Network similar to the one introduced by Kipf and Welling.
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    """
    def __init__(self, num_layers, num_input_features, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden) #GCNconv layer 6 feature-channels as input and 6 as output
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))  # remaining GCNconv layers
        self.lin1 = Linear(hidden, hidden)  #linear layer
        self.lin2 = Linear(hidden, 2)       #linear layer, output layer, 2 classes

    def reset_parameters(self):     #reset all conv and linear layers except the first GCNConv layer
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GCNConv class
        self.lin1.reset_parameters()    # .reset_parameters() is method of the torch.nn.Linear class
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #graph convolutions and relu activation
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length 6, the resulting x has shape [30, 6]
        x = global_mean_pool(x, batch)

        #linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # output = F.log_softmax(x, dim=-1)
        output = F.log_softmax(x, dim=-1) #get output between 0 and 1
        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__

class GCNWithJK(torch.nn.Module):
    # Graph Convolutional Neural Network similar to the one introduced by Kipf and Welling.
    # https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    # jumping Knowledge (Xu et al.
    def __init__(self, num_layers, num_input_features, hidden, mode='cat'):
        super(GCNWithJK, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    # https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py
    def __init__(self, num_input_features, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGEWithJK(torch.nn.Module):
    # https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py
    def __init__(self, num_input_features, num_layers, hidden, mode='cat'):
        super(GraphSAGEWithJK, self).__init__()
        self.conv1 = SAGEConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

