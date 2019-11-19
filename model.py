#!/usr/bin/python

import torch
import torch.nn.functional as F
from torch.nn import Linear, init
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, NNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, TopKPooling, global_max_pool

# own modules
from MLP import MLP

from GraphConvolutions import OwnGConv

class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural Network similar to the one introduced by Kipf and Welling.
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py

    Uses the graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper
    """
    def __init__(self, num_layers, num_input_features, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden) # GCNconv layer
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))  # remaining GCNconv layers
        self.lin1 = Linear(hidden, hidden)              # linear layer
        self.lin2 = Linear(hidden, 2)                   # linear layer, output layer, 2 classes
        self.reset_parameters()

    def reset_parameters(self):     # reset all conv and linear layers
        print("reset parameters")
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GCNConv class
        self.lin1.reset_parameters()    # .reset_parameters() is method of the torch.nn.Linear class
        self.lin2.reset_parameters()

    def forward(self, data):
        # data: Batch(batch=[num_nodes_in_batch],
        #               edge_attr=[2*num_nodes_in_batch,num_edge_features_per_edge],
        #               edge_index=[2,2*num_nodes_in_batch],
        #               pos=[num_nodes_in_batch,2],
        #               x=[num_nodes_in_batch, num_input_features_per_node],
        #               y=[num_graphs_in_batch, num_classes]
        # example: Batch(batch=[2490], edge_attr=[4980,1], edge_index=[2,4980], pos=[2490,2], x=[2490,33], y=[32,2]

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x.shape: torch.Size([num_nodes_in_batch, num_input_features_per_node])
        # edge_index.shape: torch.Size([2, 2*num_nodes_in_batch])
        # batch.shape: torch.Size([num_nodes_in_batch])
        # example:  x.shape = troch.Size([2490,33])
        #           edge_index.shape = torch.Size([2,4980])
        #           batch.shape = torch.Size([2490])

        # graph convolutions and relu activation
        x = F.relu(self.conv1(x, edge_index))
        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length hidden, the resulting x has shape [30, hidden]
        x = global_mean_pool(x, batch)
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])

        # linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape:  torch.Size([num_graphs_in_batch, num_classes)
        # example:  x.shape = torch.Size([32, 2])

        # output = F.log_softmax(x, dim=-1)
        output = F.log_softmax(x, dim=-1)           # get output between 0 and 1
        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__

class GCNWithJK(torch.nn.Module):
    """
    Graph Convolutional Neural Network similar to the one introduced by Kipf and Welling.
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    jumping Knowledge (Xu et al.)

    Uses the graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper
    """

    def __init__(self, num_layers, num_input_features, hidden, mode='cat'):
        super(GCNWithJK, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.jump = JumpingKnowledge(mode)
        if mode == 'cat': # concatenation
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
        # data: Batch(batch=[num_nodes_in_batch],
        #               edge_attr=[2*num_nodes_in_batch,num_edge_features_per_edge],
        #               edge_index=[2,2*num_nodes_in_batch],
        #               pos=[num_nodes_in_batch,2],
        #               x=[num_nodes_in_batch, num_input_features_per_node],
        #               y=[num_graphs_in_batch, num_classes]
        # example: Batch(batch=[2490], edge_attr=[4980,1], edge_index=[2,4980], pos=[2490,2], x=[2490,33], y=[32,2]

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x.shape: torch.Size([num_nodes_in_batch, num_input_features_per_node])
        # edge_index.shape: torch.Size([2, 2*num_nodes_in_batch])
        # batch.shape: torch.Size([num_nodes_in_batch])
        # example:  x.shape = troch.Size([2490,33])
        #           edge_index.shape = torch.Size([2,4980])
        #           batch.shape = torch.Size([2490])


        x = F.relu(self.conv1(x, edge_index))
        # x.shape: torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = troch.Size([2490,66])
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]  # xs: list containing layer-wise representations
        x = self.jump(xs) # aggregate information accross different layers
        # x.shape: torch.Size([num_nodes_in_batch, num_layers * hidden])
        # example: x.shape = torch.Size([2490, 2*66])

        x = global_mean_pool(x, batch)
        # x.shape: torch.Size([num_graphs_in_batch, 2*hidden])
        # example: x.shape = torch.Size([32, 2*66])

        x = F.relu(self.lin1(x))
        # x.shape: torch.Size([num_graphs_in_batch, hidden])
        # example: x.shape = torch.Size([32, 66])

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape: torch.Size([num_graphs_in_batch, num_classes])
        # example: x.shape = torch.Size([32, 2])
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    """
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py
    Uses the GraphSAGE operator from the “Inductive Representation Learning on Large Graphs” paper
    """
    def __init__(self, num_input_features, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_input_features, hidden)       # SAGEConv layer
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))         # SAGEConv layers
        self.lin1 = Linear(hidden, hidden)                      # linear layer
        self.lin2 = Linear(hidden, 2)                           # linear layer

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # data: Batch(batch=[num_nodes_in_batch],
        #               edge_attr=[2*num_nodes_in_batch,num_edge_features_per_edge],
        #               edge_index=[2,2*num_nodes_in_batch],
        #               pos=[num_nodes_in_batch,2],
        #               x=[num_nodes_in_batch, num_input_features_per_node],
        #               y=[num_graphs_in_batch, num_classes]
        # example: Batch(batch=[2490], edge_attr=[4980,1], edge_index=[2,4980], pos=[2490,2], x=[2490,33], y=[32,2]

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x.shape: torch.Size([num_nodes_in_batch, num_input_features_per_node])
        # edge_index.shape: torch.Size([2, 2*num_nodes_in_batch])
        # batch.shape: torch.Size([num_nodes_in_batch])
        # example:  x.shape = troch.Size([2490,33])
        #           edge_index.shape = torch.Size([2,4980])
        #           batch.shape = torch.Size([2490])

        x = F.relu(self.conv1(x, edge_index))
        # x.shape: torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = troch.Size([2490,66])

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        # x.shape: torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = troch.Size([2490,66])

        x = global_mean_pool(x, batch)
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])

        x = F.relu(self.lin1(x))
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape:  torch.Size([num_graphs_in_batch, num_classes)
        # example:  x.shape = torch.Size([32, 2])
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GraphSAGEWithJK(torch.nn.Module):
    """
    Uses the GraphSAGE operator from the “Inductive Representation Learning on Large Graphs” paper

    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py

    Jumping Knowledge
    """
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


class OwnGraphNN(torch.nn.Module):
    def __init__(self, num_input_features, num_layers, hidden, mode='cat'):
        super(OwnGraphNN, self).__init__()
        self.conv1 = OwnGConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(OwnGConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(2*num_layers * hidden, 2*hidden)
        else:
            self.lin1 = Linear(hidden, 2*hidden)
        self.lin2 = Linear(2*hidden, hidden)
        self.lin3 = Linear(hidden, 2)
        # self.bn_conv1 = torch.nn.BatchNorm1d(hidden)
        # self.bn_conv2 = torch.nn.BatchNorm1d(hidden)

        self.reset_parameters()


    def reset_parameters(self):
        print("reset params")
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        # init.xavier_uniform_(self.lin1.weight, gain=init.calculate_gain("relu"))
        # init.xavier_uniform_(self.lin2.weight, gain=init.calculate_gain("relu"))
        # init.xavier_uniform_(self.lin3.weight, gain=init.calculate_gain("relu"))

        # init.constant_(self.lin1.bias, 0.01)
        # init.constant_(self.lin2.bias, 0.01)
        # self.bn_conv1.reset_parameters()
        # self.bn_conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # graph conv, ReLU non linearity, dropout
        # x = F.relu(self.bn_conv1(self.conv1(x, edge_index)))
        # x = F.relu(self.conv1(x, edge_index))
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5,  training=self.training) #################
        xs = [x]

        # graph convs, ReLU non linearity, dropout
        for conv in self.convs:
            # x = F.relu(self.bn_conv2(conv(x, edge_index)))
            # x = F.relu(conv(x, edge_index))
            x = conv(x, edge_index)
            x = F.dropout(x, p=0.5, training=self.training) #################
            xs += [x]

        x = self.jump(xs)

        # graph pooling
        # x = global_mean_pool(x, batch)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1) ##################

        # linear layer, ReLU non linearity, dropout
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # linear layer, ReLU non linearity, dropout
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # final linear layer, log_softmax
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GraphNN(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, hidden):
        super(GraphNN, self).__init__()
        self.conv1 = GraphConv(num_input_features, hidden)
        self.pool1 = TopKPooling(hidden, ratio=1)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))

        self.pools = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.pools.append(TopKPooling(hidden, ratio=1))

        self.lin1 = Linear(2*hidden, hidden)
        self.lin2 = Linear(hidden, 2)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        xs = [torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)]

        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
            xs.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))
        x = sum(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class NMP(torch.nn.Module):
    """
    Graph Convolutional Neural Network similar to the one introduced by Kipf and Welling.
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    """
    def __init__(self, num_layers, num_input_features, hidden, nn):
        super(NMP, self).__init__()
        self.conv1 = NNConv(num_input_features, hidden, nn, aggr="add")
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(NNConv(hidden, hidden, nn, aggr="add"))
        self.lin1 = Linear(hidden, hidden)  #linear layer
        self.lin2 = Linear(hidden, 2)       #linear layer, output layer, 2 classes
        self.reset_parameters()

    def reset_parameters(self):     #reset all conv and linear layers except the first GCNConv layer
        print("reset parameters")
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GCNConv class
        self.lin1.reset_parameters()    # .reset_parameters() is method of the torch.nn.Linear class
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        #graph convolutions and relu activation
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

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