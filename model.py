import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural Network similar to the one introduced by Kipf et al.
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    """
    def __init__(self, dataset, num_layers, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(6, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 2)

    def reset_parameters(self):
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


