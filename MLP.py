#!/usr/bin/python

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, n_in, n_out, h_layers=(20,20)):
        super(MLP, self).__init__()

        self.n_hlayers = len(h_layers) # get the number of hidden layers

        # initialize all fully connected layers
        self.fcs = nn.ModuleList([nn.Linear(n_in, h_layers[i]) if i == 0 else # input layer
                                  nn.Linear(h_layers[i - 1], n_out) if i == self.n_hlayers else # output layer
                                  nn.Linear(h_layers[i - 1], h_layers[i]) for i in range(self.n_hlayers + 1)]) # hidden layers

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_flat_features(x))
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features