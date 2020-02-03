#!/usr/bin/python

import torch
import torch.nn.functional as F
from torch.nn import Linear, init
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, NNConv, GATConv, GINConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, TopKPooling, global_max_pool, GlobalAttention, global_add_pool, Set2Set

# own modules
from GraphConvolutions import OwnGConv, OwnGConv2




###################################################################################################
###################################################################################################


class GCN(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden) # GCNconv layer
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))  # remaining GCNconv layers

        self.lin1 = Linear(3*hidden, hidden)             # linear layer
        self.lin2 = Linear(hidden, 2)

    def reset_parameters(self):     # reset all conv and linear layers
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################

        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length hidden, the resulting x has shape [30, hidden]

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch),global_max_pool(x, batch)], dim=1)
        # x.shape:  torch.Size([num_graphs_in_batch, 3*hidden)
        # example:  x.shape = torch.Size([32, 3*66])

        # linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape:  torch.Size([num_graphs_in_batch, num_classes)
        # example:  x.shape = torch.Size([32, 2])

        output = F.log_softmax(x, dim=-1)

        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__


###################################################################################################
###################################################################################################


class GCNWithJK(torch.nn.Module):


    def __init__(self, num_layers, num_input_features, hidden, mode='cat'):
        super(GCNWithJK, self).__init__()
        self.conv1 = GCNConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.jump = JumpingKnowledge(mode)
        if mode == 'cat': # concatenation
            self.lin1 = Linear(3*num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(3*hidden, hidden)
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################
            xs += [x]  # xs: list containing layer-wise representations
        x = self.jump(xs) # aggregate information across different layers (concatenation)
        # x.shape: torch.Size([num_nodes_in_batch, num_layers * hidden])
        # example: x.shape = torch.Size([2490, 2*66])

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
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
        # return F.softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


###################################################################################################
###################################################################################################


class GraphSAGE(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_input_features, hidden)       # SAGEConv layer
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))         # SAGEConv layers
        self.lin1 = Linear(3*hidden, hidden)                      # linear layer
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
        # x = F.dropout(x, p=0.2, training=self.training) ##################


        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x.shape: torch.Size([num_nodes_in_batch, hidden])
            # example:  x.shape = troch.Size([2490,66])
            # x = F.dropout(x, p=0.2, training=self.training)  ##################

        
        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch),global_max_pool(x, batch)], dim=1)
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


###################################################################################################
###################################################################################################


class GraphSAGEWithJK(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, hidden, mode='cat'):
        super(GraphSAGEWithJK, self).__init__()
        self.conv1 = SAGEConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(3*num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(3*hidden, hidden)
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################
            xs += [x]
        x = self.jump(xs)

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


###################################################################################################
###################################################################################################


class GATNet(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, hidden):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(num_input_features, hidden) # GATconv layer
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden))  # remaining GATconv layers
        self.lin1 = Linear(3*hidden, hidden)              # linear layer
        self.lin2 = Linear(hidden, 2)                   # linear layer, output layer, 2 classes

    def reset_parameters(self):     # reset all conv and linear layers
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GATConv class
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################
        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length hidden, the resulting x has shape [30, hidden]
        # x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
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

        output = F.log_softmax(x, dim=-1)
        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__


###################################################################################################
###################################################################################################

###################################################################################################
###################################################################################################


class NMP(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, hidden, nn):
        super(NMP, self).__init__()
        self.conv1 = NNConv(num_input_features, hidden, nn(1, num_input_features*hidden), aggr="add")
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(NNConv(hidden, hidden, nn(1, hidden*hidden), aggr="add"))

        self.lin1 = Linear(3*hidden, hidden)  # linear layer
        self.lin2 = Linear(hidden, 2)       # linear layer, output layer, 2 classes

    def reset_parameters(self):     # reset all conv and linear layers except the first GCNConv layer
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GCNConv class
        self.lin1.reset_parameters()    # .reset_parameters() is method of the torch.nn.Linear class
        self.lin2.reset_parameters()
        self.att.reset_parameters()
        # self.set2set.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # graph convolutions and relu activation
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length 6, the resulting x has shape [30, 6]

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        #linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        output = F.log_softmax(x, dim=-1) #get output between 0 and 1

        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__


###################################################################################################
###################################################################################################


class GIN(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(Linear(num_input_features, hidden), torch.nn.ReLU(), Linear(hidden, hidden), torch.nn.ReLU()),
                             num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINConv(torch.nn.Sequential(Linear(hidden, hidden), torch.nn.ReLU(), Linear(hidden, hidden), torch.nn.ReLU()),hidden, hidden))  # remaining GCNconv layers

        self.lin1 = torch.nn.Linear(3*hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, 2)

    def reset_parameters(self):     # reset all conv and linear layers
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################

        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length hidden, the resulting x has shape [30, hidden]

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch),global_max_pool(x, batch)], dim=1)
        # x.shape:  torch.Size([num_graphs_in_batch, 3*hidden)
        # example:  x.shape = torch.Size([32, 3*66])

        # linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape:  torch.Size([num_graphs_in_batch, num_classes)
        # example:  x.shape = torch.Size([32, 2])

        output = F.log_softmax(x, dim=-1)

        return output

    def __repr__(self):
        #for getting a printable representation of an object
        return self.__class__.__name__


###################################################################################################
###################################################################################################

class GraphNN(torch.nn.Module):

    def __init__(self, num_layers, num_input_features, hidden):
        super(GraphNN, self).__init__()
        self.conv1 = GraphConv(num_input_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))

        self.lin1 = torch.nn.Linear(3*hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, 2)

    def reset_parameters(self):  # reset all conv and linear layers
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()  # .reset_parameters() is method of the torch_geometric.nn.GCNConv class
        self.lin1.reset_parameters()  # .reset_parameters() is method of the torch.nn.Linear class
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
        # x = F.dropout(x, p=0.5, training=self.training)  ##################

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            # x = F.dropout(x, p=0.5, training=self.training)  ##################

        # x.shape:  torch.Size([num_nodes_in_batch, hidden])
        # example:  x.shape = torch.Size([2490, 66])

        # global mean pooling: the feature vector of every node of one graph are summed and the mean is taken
        # if there are 30 graphs in the batch and the feature vector has length hidden, the resulting x has shape [30, hidden]

        x = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # x.shape:  torch.Size([num_graphs_in_batch, 3*hidden)
        # example:  x.shape = torch.Size([32, 3*66])

        # linear layers, activation function, dropout and softmax
        x = F.relu(self.lin1(x))
        # x.shape:  torch.Size([num_graphs_in_batch, hidden)
        # example:  x.shape = torch.Size([32, 66])
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # x.shape:  torch.Size([num_graphs_in_batch, num_classes)
        # example:  x.shape = torch.Size([32, 2])

        output = F.log_softmax(x, dim=-1)

        return output

    def __repr__(self):
        # for getting a printable representation of an object
        return self.__class__.__name__
###################################################################################################
###################################################################################################
