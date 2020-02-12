#!/usr/bin/python

import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected

import os
import csv
import re
from pandas import read_excel
import numpy as np
import random
from statistics import stdev, mean

class DataConstructor():
    def __init__(self):
        pass

    def get_graph(selfs, folder, filename, raw = False , k=0):

        # initialize tree from paper-graph dataset to get edge information (because base dataset doesnt has edges)
        tree = ET.ElementTree(file= "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
        root = tree.getroot()

        # get edge index
        ##############################
        # get the start and end points of every edge and store them in a list of lists
        start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root.iter("edge")]
        end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root.iter("edge")]
        edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        # create a tensor needed to construct the graph
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        # print(edge_index.size())
        edge_index = to_undirected(edge_index.t().contiguous())
        # print(edge_index.size())
        img = re.split("_", filename)[0]
        name = re.split("\.", filename)[0]

        # get node features and position
        ##############################
        # initialize the list
        all_node_features = []
        all_node_positions = []

        if not raw:
            with open("pT1_dataset/Mean_and_Sd_k" + str(k) + ".csv", mode="r") as file:
                reader = csv.reader(file, delimiter=",")
                header = next(reader)
                mean_vec = np.asarray(next(reader))[1:].astype(float)
                sd_vec = np.asarray(next(reader))[1:].astype(float)

        # open the csv file containing the raw feature values of the graph extracted from filename
        raw_data = read_excel("pT1_dataset/dataset/" + img +"/" + name + "/" + name + "-features.xlsx", header=0)

        for index, row in raw_data.iterrows():

            if folder == "pT1_dataset/graphs/base-dataset/":
                feature_vec = [float(f) for f in row[4:]]
                node_position = [float(p) for p in row[2:4]]

                if not raw:     # normalize the data using the mean and sd of the training set
                    feature_vec = np.asarray(feature_vec)
                    feature_vec = list((feature_vec - mean_vec[3:]) / sd_vec[3:])
                    node_position = np.asarray(node_position)
                    node_position = list((node_position - mean_vec[1:3]) / sd_vec[1:3])


            if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
                feature_vec = [float(f) for f in [row[10], row[13], row[14], row[35]]]
                node_position = [float(p) for p in row[2:4]]

                if not raw:     # normalize the data using the mean and sd of the training set
                    feature_vec = np.asarray(feature_vec)
                    feature_vec = list((feature_vec - mean_vec[[9,12,13,34]]) / sd_vec[[9,12,13,34]])
                    node_position = np.asarray(node_position)
                    node_position = list((node_position - mean_vec[1:3]) / sd_vec[1:3])

            # append the feature vec and position of the current node to the list
            all_node_features.append(feature_vec)
            all_node_positions.append(node_position)

        x = torch.tensor(all_node_features, dtype=torch.float)      # create a tensor needed to construct the graph
        pos = torch.tensor(all_node_positions, dtype=torch.float)   # create a tensor needed to construct the graph

        # get the label of the class
        ###########################
        if "abnormal" in filename:
            graph_class = [1,0]
            cls = 1
        elif "normal" in filename:
            graph_class = [0,1]
            cls = 0
        else:
            print("the filename has not the correct format")
        y = torch.tensor([graph_class], dtype=torch.float)          # create a tensor needed to construct the graph


        # get the edge feature (length of the edge based on the node coordinates)
        ############################
        distances = []
        for i in range(len(edge_index[0])):             # iterate over every edge
            positions = pos[[edge_index[0,i].item(), edge_index[1,i].item()]]           # get the coordinates of the nodes connected by the edge
            # distances.append([torch.dist(positions[0], positions[1], p=2)])         # compute L2 norm  (if distances are not normalized use square brackets directly)

            distances.append(torch.dist(positions[0], positions[1], p=2))       # compute L2 norm; distances is now a list of tensors
        # # normalize distances
        # avg = torch.mean(torch.stack(distances))        # compute the mean for all tensors in the list
        # stdv = torch.std(torch.stack(distances))        # compute the standard deviation for all tensors in the list

        if not raw:
            distances = [(dist-mean_vec[0])/sd_vec[0] for dist in distances]         # normalization
        distances = [[item] for item in distances]                  # every tensor is packed into its own list
        edge_attr = torch.tensor(distances, dtype=torch.float)      # create a tensor needed to construct the graph

        img_patch = [int(i) for i in re.findall(r"[0-9]+", filename)]
        img_patch.append(cls)
        graph_name = torch.tensor([img_patch], dtype=torch.float)

        # construct the graph
        #############################
        graph = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index, pos=pos, name=graph_name)

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

        for file in sorted(os.listdir(folder)):         # iterate over every file in the folder
            if file.endswith(".gxl"):           # check if its a gxl file
                r_file_names.append(file)       # add the filename to the list

        return r_file_names                     # return the list of gxl filenames

    def get_data_list(self, folder, k=0, raw=True):
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
                data = self.get_graph(folder, file, raw = raw, k=k)
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


    def compute_mean_sd_per_split(self):
        """
        for every train data split the mean and the standard deviation of every attribute (node, edge and position) is
        computed and saved (one csv file per data split)
        """
        avg_f = [[], [], [], []]
        stdv_f = [[], [], [], []]
        avg_pos = [[], [], [], []]
        stdv_pos = [[], [], [], []]
        avg_e = [[], [], [], []]
        stdv_e = [[], [], [], []]

        for k in range(4):      # iterate over the 4 different data splits
            all_node_features = []
            all_positions = []
            all_edge_features = []

            # get the train data list
            train_data_list, val_data_list, test_data_list = self.get_data_list("pT1_dataset/graphs/base-dataset/", k=k, raw=True)
            for gidx in range(len(train_data_list)):    # iterate over every graph in the training set
                for nidx in range(len(train_data_list[gidx])):   # iterate over every node of the graph
                    all_node_features.append(train_data_list[gidx].x[nidx])     # save the node features

                all_positions.append(train_data_list[gidx].pos)                 # save the node positions
                all_edge_features.append(train_data_list[gidx].edge_attr)       # save the edge features

            avg_f[k]= torch.mean(torch.stack(all_node_features), dim=0)         # compute the mean for every feature
            stdv_f[k] = torch.std(torch.stack(all_node_features), dim=0)        # compute sd for every feature

            avg_pos[k] = torch.mean(torch.cat(all_positions), dim=0)            # compute mean across all node positions
            stdv_pos[k] = torch.std(torch.cat(all_positions), dim=0)            # compute the sd across all node positions

            avg_e[k] = torch.mean(torch.cat(all_edge_features))                 # compute mean of the edge feature
            stdv_e[k] = torch.std(torch.cat(all_edge_features))                 # compute sd of edge feature

            with open("pT1_dataset/Mean_and_Sd_k" + str(k) + ".csv", "w", newline="") as file:      # create csv file for specific data split
                csv_writer = csv.writer(file)
                # write header of the csv file
                headers = [["metric"], ["edge_attr"], ["x_coordinate"], ["y_coordinate"], ["f%s" % i for i in range(4, 37)]]
                headers = [item for lst in headers for item in lst]
                csv_writer.writerow(headers)
                # write rows of the csv file
                means = [["mean"], [avg_e[k].item()], [avg_pos[k][0].item()], [avg_pos[k][1].item()], [f.item() for f in avg_f[k]]]
                sds = [["sd"], [stdv_e[k].item()], [stdv_pos[k][0].item()], [stdv_pos[k][1].item()], [f.item() for f in stdv_f[k]]]
                csv_writer.writerow([item for lst in means for item in lst])
                csv_writer.writerow([item for lst in sds for item in lst])

if __name__ == "__main__":
    folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "pT1_dataset/graphs/base-dataset/"
    filename = "img0_0_normal.gxl"

    raw_data= DataConstructor() # initialize instance

    # make csv files containing the mean and sd of all attributes
    ##################################
    # raw_data.compute_mean_sd_per_split()

    num_nd_ab = []
    num_nd_n = []
    for k in range(1):
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder, raw=False, k=k) # get data lists for train, val and test
        print("number of graphs",len(train_data_list))         # how many graphs are inside the train_data_list
        print("first graph:",train_data_list[0])           # data object for first graph
        print("graph from:", train_data_list[0].name)
        print("feature vec:",train_data_list[0].x[0])   # feature vec of first node
        print("data object for graph of img_0_0_normal:", raw_data.get_graph(folder, filename, raw=False, k=k))

        # Dataloader
        # train_data = DataLoader(train_data_list, batch_size=32)
        # for batch in train_data:
        #     print("num of graphs in batch:", batch.num_graphs)
        #     print("batch:", batch)
        #     break

        print("dimension of edge_attr:", train_data_list[0].edge_attr.dim())
        # print("edge_attr of first graph", train_data_list[0].edge_attr)

        import matplotlib.pyplot as plt
        import numpy as np
        from statistics import stdev, mean

        # train_data_list = test_data_list
        # train_data_list = val_data_list

        # plot feature distributions (normal and abnormal glands together)
        # plt.rc("font", size=5)  # change font size
        #
        # for f_idx in range(train_data_list[0].num_node_features):   # get indices of every feature
        #     f_vec = []                                              # f_vec will contain the values of a particular feature of every node and graph
        #
        #     for graph in train_data_list:                           # iterate over every graph in train_data_list
        #         for nd_idx in range(graph.num_nodes):               # get indices of every node
        #             f_vec.append(graph.x[nd_idx][f_idx].item())
        #
        #     plt.subplot(4, len(train_data_list[0].x[0])//4+1, f_idx+1)
        #     plt.hist(f_vec, density=True)
        #     plt.title("f"+ str(f_idx+4) + "  std: " + str(torch.std(torch.tensor(f_vec)).item())[:5] + "  mean: " + str(torch.mean(torch.tensor(f_vec)).item())[:5])

        # plt.tight_layout()
        # plt.show()

        # plot feature distributions (normal vs abnormal glands)
        # for f_idx in range(train_data_list[0].num_node_features):
        #     f_vec_normal = []
        #     f_vec_abnormal = []
        #
        #     for graph in train_data_list:
        #
        #         if graph.y[0][0].item() == 1:  # abnormal
        #             for nd_idx in range(graph.num_nodes):
        #                 f_vec_abnormal.append(graph.x[nd_idx][f_idx])
        #
        #         elif graph.y[0][0].item() == 0: #normal
        #             for nd_idx in range(graph.num_nodes):
        #                 f_vec_normal.append(graph.x[nd_idx][f_idx])
        #
        #     plt.subplot(4, train_data_list[0].num_node_features//4+1, f_idx+1)
        #     plt.hist(f_vec_normal, alpha=0.5, label="n", density=True)
        #     plt.hist(f_vec_abnormal, alpha=0.5, label="a", density=True)
        #     plt.legend()
        #     plt.title("f"+ str(f_idx+4))
        # plt.tight_layout()
        # plt.show()

        # plot number of nodes (normal vs abnormal glands)

        for data_list in [train_data_list, val_data_list, test_data_list]:
            for graph in data_list:
                if graph.y[0][0].item() == 1:
                    num_nd_ab.append(graph.num_nodes)
                if graph.y[0][0].item() == 0:
                    num_nd_n.append(graph.num_nodes)

    print("max abnormal:", np.max(np.asarray(num_nd_ab)))
    print("min abnormal:", np.min(np.asarray(num_nd_ab)))
    print("max_normal:", np.max(np.asarray(num_nd_n)))
    print("min_normal:", np.min(np.asarray(num_nd_n)))
    print("median _abnormal:", np.median(np.asarray(num_nd_ab)))
    print("median_normal:", np.median(np.asarray(num_nd_n)))
    for i in num_nd_ab:
        num_nd_n.append(i)
    print("median total:", np.median(np.asarray(num_nd_n)))
        # bins = np.linspace(0,600, 20)

        # plt.hist(num_nd_n, bins, alpha=0.5, label="normal")
        # plt.hist(num_nd_ab, bins, alpha=0.5, label="abnormal")
        # plt.title("number of nodes per class")
        # plt.ylabel("freq")
        # plt.legend()
        # plt.show()