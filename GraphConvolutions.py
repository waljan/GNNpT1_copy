import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, dropout
from torch_geometric.nn import inits
from torch.nn import init



class OwnGConv(MessagePassing):
    """
    old node embedding xi
    new node embedding x'i
    neighboring node xj         because self loops are added, xi is also considered a neighbor
    learnable weight matrices W
    non-linearity act

    message: mean(ReLU(W1*xj))
    update : ReLU(W2*concat(message, xi))

    x'i =  ReLU( W2*(concat(mean(RelU(W1*xj)), xi) )


    """
    def __init__(self, in_channels, out_channels):
        super(OwnGConv, self).__init__(aggr="mean")
        self.m_lin = torch.nn.Linear(in_channels, out_channels)
        self.u_lin = torch.nn.Linear(out_channels + in_channels, out_channels)
        self.act = torch.nn.ReLU()

        # self.m_weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        # self.u_weight = torch.nn.Parameter(torch.Tensor(out_channels + in_channels, out_channels))
        # self.m_bias = torch.nn.Parameter(torch.Tensor(out_channels))
        # self.u_bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.m_lin.reset_parameters()
        self.u_lin.reset_parameters()

        # inits.glorot(self.m_lin.weight)
        # init.constant_(self.m_lin.bias, 0.01)
        # inits.glorot(self.u_lin.weight)
        # init.constant_(self.u_lin.bias, 0.01)
    #     init.xavier_uniform_(self.m_lin.weight, gain=init.calculate_gain("relu"))
    #     init.xavier_uniform_(self.u_lin.weight, gain=init.calculate_gain("relu"))
    #     init.constant_(self.m_lin.bias, 0.01)
    #     init.constant_(self.u_lin.bias, 0.01)

    def forward(self, x, edge_index):

        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # TODO: test if it performs better when using self loops

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.m_lin(x_j)
        x_j = self.act(x_j)         # TODO: test if ReLU is need here (does it improve the model, does it even make sense?)
        # x_j = torch.nn.functional.dropout(x_j, p=0.5, training=self.training)

        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.u_lin(new_embedding)
        new_embedding = self.act(new_embedding)
        # new_embedding = torch.nn.functional.dropout(new_embedding, p=0.5, training=self.training)

        # return new_embedding
        return new_embedding



class OwnGConv2(MessagePassing):
    """
    try to use edge features

    message: h(xj * eij)
    """

    def __init__(self, in_channels, out_channels):
        super(OwnGConv, self).__init__(aggr="mean")
        self.m_lin = torch.nn.Linear(in_channels, out_channels)
        self.u_lin = torch.nn.Linear(out_channels + in_channels, out_channels)
        self.act = torch.nn.ReLU()

        # self.m_weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        # self.u_weight = torch.nn.Parameter(torch.Tensor(out_channels + in_channels, out_channels))
        # self.m_bias = torch.nn.Parameter(torch.Tensor(out_channels))
        # self.u_bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.m_lin.reset_parameters()
        self.u_lin.reset_parameters()

        # inits.glorot(self.m_lin.weight)
        # init.constant_(self.m_lin.bias, 0.01)
        # inits.glorot(self.u_lin.weight)
        # init.constant_(self.u_lin.bias, 0.01)
    #     init.xavier_uniform_(self.m_lin.weight, gain=init.calculate_gain("relu"))
    #     init.xavier_uniform_(self.u_lin.weight, gain=init.calculate_gain("relu"))
    #     init.constant_(self.m_lin.bias, 0.01)
    #     init.constant_(self.u_lin.bias, 0.01)

    def forward(self, x, edge_index, edge_attr):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, in_channels]
        x_j = self.m_lin(x_j)
        x_j = self.act(x_j)
        # x_j = torch.nn.functional.dropout(x_j, p=0.5, training=self.training)

        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.u_lin(new_embedding)
        new_embedding = self.act(new_embedding)
        # new_embedding = torch.nn.functional.dropout(new_embedding, p=0.5, training=self.training)

        # return new_embedding
        return new_embedding
