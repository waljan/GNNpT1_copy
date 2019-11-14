#!/usr/bin/python

import copy
import numpy as np

def augment(train_data_list, n=4):
    """
    augments the data set by randomly adding small values (between 0 and 1) to every node

    data object representing a graph:
    Data(edge_attr=[n_Nodes, 2*n_edge_attr], edge_index=[2, 2*n_Nodes], pos=[N, 2], x=[n_Nodes, n_node_features], y=[1, num_classes])
    e.g.: Data(edge_attr=[224, 1], edge_index=[2, 224], pos=[112, 2], x=[112, 33], y=[1, 2])

    :param train_data_list: list of graph data objects
    :param n: factor by which the dataset will be increased
    :return: train_data_list_aug: list of graph data objects containing the original and the changed graphs
    """
    train_data_list_aug = copy.deepcopy(train_data_list)
    c = 0
    for i in range(n):  # how often the augmentation is done
        r_factors = np.random.rand(101, 101)  # create array of random numbers
        train_data_list_cp = copy.deepcopy(train_data_list)
        for graph in range(len(train_data_list)):  # iterate over every graph
            for nd in range(len(train_data_list[graph].x)):  # iterate over every node
                for f in range(len(train_data_list[graph].x[nd])):  # iterate over every node feature
                    choice = r_factors[f, c]  # draw random number to determine whether to add or subtract
                    r_factor = r_factors[
                        c + 1, f]  # draw a random number to determine the value that will be added/subtracted
                    r_factor2 = r_factors[c, c + 1]
                    c += 1
                    if c == 100:
                        c = 0
                    if choice >= 0.5:
                        train_data_list_cp[graph].x[nd][
                            f] += r_factor2 * r_factor  # add a small random value to the feature "f" of node "nd" of graph "graph"
                    if choice < 0.5:
                        r_factor = 1 - r_factor
                        train_data_list_cp[graph].x[nd][
                            f] -= r_factor2 * r_factor  # subtract a small random value to the feature "f" of node "nd" of graph "graph"

            train_data_list_aug.append(train_data_list_cp[graph])

    return train_data_list_aug