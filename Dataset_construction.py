#!/usr/bin/python

import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import os
import csv
import re
import random

class DataConstructor():
    def __init__(self):
        pass
    def get_graph(self, folder, filename):
        """
        this function takes a gxl-file as input and creates a graph as it is used in pytorch Geometric
        :param filename: gxl file that stores a graph
        :param folder:
        :return python object modeling a single graph with various attributes
        """
        # initialize tree from chosen folder/file
        tree = ET.ElementTree(file=folder + filename)
        root = tree.getroot()

        # get edges
        ##############################
        # initialize tree from paper-graph dataset to get edge information (because base dataset doesnt has edges)
        tree2 = ET.ElementTree(file= "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
        root2 = tree2.getroot()

        # get the start and end points of every edge and store them in a list of lists
        start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root2.iter("edge")]
        end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root2.iter("edge")]
        edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        # create a tensor needed to construct the graph
        edge_index = torch.tensor(edge_list, dtype=torch.long)

        # get node features and position
        ##############################
        # initialize the list
        all_node_features = []
        all_node_positions = []

        for node in root.iter("node"):        # iterate over every node
            feature_vec = [float(value.text) for feature in node for value in feature]      # get the feature vector of every node

            if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
                node_position = feature_vec[4:]         # the last two entries corespond to the node-coordinates in the image
                feature_vec = feature_vec[:4]           # the coordinates are not used as features


            if folder == "pT1_dataset/graphs/base-dataset/":
                node_position = feature_vec[33:]        # the last two entries corespond to the node-coordinates in the image
                feature_vec = feature_vec[:33]          # the coordinates are not used as features

            # append the feature vec and position of the current node to the list
            all_node_features.append(feature_vec)
            all_node_positions.append(node_position)

        x = torch.tensor(all_node_features, dtype=torch.float)      # create a tensor needed to construct the graph
        pos = torch.tensor(all_node_positions, dtype=torch.float)   # create a tensor needed to construct the graph

        # get the label of the class
        ###########################
        if "abnormal" in filename:
            graph_class = [1,0]
        elif "normal" in filename:
            graph_class = [0,1]
        else:
            print("the filename has not the correct format")
        y = torch.tensor([graph_class], dtype=torch.float)          # create a tensor needed to construct the graph


        # get the edge feature (length of the edge based on the node coordinates)
        ############################
        distances = []
        for edge in edge_index:             # iterate over every edge
            positions = pos[edge]           # get the coordinates of the nodes connected by the edge
            distances.append([torch.dist(positions[0], positions[1], p=2)])         # compute L2 norm

        edge_attr = torch.tensor(distances, dtype=torch.float)      # create a tensor needed to construct the graph

        # construct the graph
        #############################
        graph = Data(x=x, y=y, edge_attr =edge_attr, edge_index=edge_index.t().contiguous(), pos=pos)


        return (graph)

    def split(self, k):
        """
        This function splits the pt1 graph data set into train, validation and test
        according to the file dataset_split.csv. This file contains the information about
        how to split the data in a 4-fold cross validation.

        :param k: integer between 0 and 3 (which of the 4 possible splits in the 4-fold cross validation)
        :return: dictionary that has the img (e.g. img6) as key and "train", "val" or "test" as value
        """

        # open the csv file
        with open("pT1_dataset/dataset_split.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            dic = {}

            # iterate over every row of the csv file
            for row in csv_reader:
                if line_count == 0:         # ignore the header
                    line_count += 1
                else:                       # use a dictionary to save the information about how to split the data into train test and val
                    dic[row[0]] = [row[k+1]]
        return dic


    def raw_file_names(self, folder):
        """
        this function creates a list of all the gxl filenames located in the directory "folder"
        :param folder: defines the folder in which the gxl files are located
        :return: list of gxl-filenames
        """

        r_file_names = []

        for file in os.listdir(folder):         # iterate over every file in the folder
            if file.endswith(".gxl"):           # check if its a gxl file
                r_file_names.append(file)       # add the filename to the list

        return r_file_names                     # return the list of gxl filenames

    def get_data_list(self, folder, k=0):
        """
        this function creates lists of data objects (data lists) separated into train val and test
        :param k: k can take values from 0 to 3 and defines which datasplit in the 4-fold cross validation is used
        :param folder: determines which dataset is used (paper or base dataset)
        :return: lists of data objects for train val and test
        """

        data_split = self.split(k) # get dictionary that tells how to split the data into train val and test

        # initialize list
        train_data_list = []
        test_data_list = []
        val_data_list = []

        pattern = "_"
        a=0

        for file in self.raw_file_names(folder):        # iterate over every filename

            # create the data object representing the graph stored in the file
            try:
                data = self.get_graph(folder, file)
            except:
                print(file, "could not be loaded")
                continue
            img = re.split(pattern, file)[0]            # get the image from which this graph was sampled
            split = data_split[img][0]                  # determine whether this graph belongs to train val or test

            # add the data object for the graph to the corresponding list
            if split == "test":
                test_data_list.append(data)
            elif split == "val":
                val_data_list.append(data)
            elif split == "train":
                train_data_list.append(data)

        return train_data_list, val_data_list, test_data_list





if __name__ == "__main__":
    folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "pT1_dataset/graphs/base-dataset/"
    filename = "img0_0_normal.gxl"

    raw_data= DataConstructor() # initialize instance
    train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder) # get data lists for train, val and test
    print(len(train_data_list))
    print(train_data_list[0])
    print("feature vec:",train_data_list[0].x[0])
    print(raw_data.get_graph(folder, filename))

    # Dataloader
    train_data = DataLoader(train_data_list, batch_size=32)
    for batch in train_data:
        print(batch.num_graphs)
        print(batch)


    import matplotlib.pyplot as plt
    import numpy as np
    from statistics import stdev, mean


    # plot feature distributions (normal and abnormal glands together)
    plt.rc("font", size=5)  # change font size

    for f in range(len(train_data_list[0].x[0])):
        f_vec = []

        for graph in train_data_list:
            for nd in range(len(graph.x)):
                f_vec.append(graph.x[nd][f].item())

        plt.subplot(4, len(train_data_list[0].x[0])//4+1, f+1)
        plt.hist(f_vec, density=True)
        plt.title("f"+ str(f) + "  std: " + str(stdev(f_vec))[:4] + "  mean: " + str(mean(f_vec))[:4])

    # plt.tight_layout()
    plt.show()

    # plot feature distributions (normal vs abnormal glands)
    for f in range(len(train_data_list[0].x[0])):
        f_vec_normal = []
        f_vec_abnormal = []

        for graph in train_data_list:

            if graph.y[0][0].item() == 1:  # abnormal
                for nd in range(len(graph.x)):
                    f_vec_abnormal.append(graph.x[nd][f])

            elif graph.y[0][0].item() == 0: #normal
                for nd in range(len(graph.x)):
                    f_vec_normal.append(graph.x[nd][f])

        plt.subplot(4, len(train_data_list[0].x[0])//4+1, f+1)
        plt.hist(f_vec_normal, alpha=0.5, label="n", density=True)
        plt.hist(f_vec_abnormal, alpha=0.5, label="a", density=True)
        plt.legend()
        plt.title("f"+ str(f))
    plt.tight_layout()
    plt.show()

    # plot number of nodes (normal vs abnormal glands)
    num_nd_ab = []
    num_nd_n = []
    for graph in train_data_list:
        if graph.y[0][0].item() == 1:
            num_nd_ab.append(graph.num_nodes)
        if graph.y[0][0].item() == 0:
            num_nd_n.append(graph.num_nodes)
    bins = np.linspace(0,600, 20)
    plt.hist(num_nd_n, bins, alpha=0.5, label="normal")
    plt.hist(num_nd_ab, bins, alpha=0.5, label="abnormal")
    plt.title("number of nodes per class")
    plt.ylabel("freq")
    plt.legend()
    plt.show()