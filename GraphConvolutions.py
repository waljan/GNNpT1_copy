import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class OwnGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(OwnGraphConv, self).__init__(aggr="mean")
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)
        self.act = torch.nn.ReLU()


    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        return self.porpagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.lin(x_j)
        return x_j

    def update(self, aggr_out, x):
        pass
        # return new_embedding

