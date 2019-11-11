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
        :return python object modeling a single graph with various attributes
        """
        # initialize tree
        tree = ET.ElementTree(file=folder + filename)
        root = tree.getroot()

        tree2 = ET.ElementTree(file= "graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
        root2 = tree2.getroot()

        # get the start and end points of every edge and store them in a list of lists
        start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root2.iter("edge")]
        end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root2.iter("edge")]
        edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        # create a tensor needed to construct the graph
        edge_index = torch.tensor(edge_list, dtype=torch.long)

        # initialize the list
        all_node_features = []
        all_node_positions = []
        # iterate over every node
        for node in root.iter("node"):
            # get the feature vector of every node
            feature_vec = [float(value.text) for feature in node for value in feature]

            if folder == "graphs/paper-graphs/distance-based_10_13_14_35/":
                node_position = feature_vec[4:]
                feature_vec = feature_vec[:4]   # the last two entries corespond to the node-coordinates in the image
                                                # the coordinates are not used as features

            if folder == "graphs/base-dataset/":
                node_position = feature_vec[33:]
                feature_vec = feature_vec[:33]

            # and append it to the list
            all_node_features.append(feature_vec)
            all_node_positions.append(node_position)

        # create a tensor needed to construct the graph

        x = torch.tensor(all_node_features, dtype=torch.float)

        # add graph label
        if "abnormal" in filename:
            graph_class = [1,0]
        elif "normal" in filename:
            graph_class = [0,1]
        else:
            print("the filename has not the correct format")
        y = torch.tensor([graph_class], dtype=torch.float)

        pos = torch.tensor(all_node_positions, dtype=torch.float)
        # construct the graph
        graph = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), pos=pos)


        return (graph)

    def split(self, k):
        """
        This function splits the pt1 graph data set into train validation and test
        according to the file dataset_split.csv. This file contains the information about
        how to split the data in a 4-fold cross validation.

        :param k: integer between 0 and 3 (which fold of the 4-fold cross validation)
        :return: dictionary that has the img (e.g. img6) as key and "train", "val" or "test" as value
        """

        # open the csv file
        with open("dataset_split.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            dic = {}

            # iterate over every row of the csv file
            for row in csv_reader:
                # ignore the header
                if line_count == 0:
                    line_count += 1
                # use a dictionary to save the information about how to split the data into train test and val
                else:
                    dic[row[0]] = [row[k+1]]
        return dic

    # get a list of all filenames
    def raw_file_names(self, folder):
        """
        this function creates a list of all the gxl filenames in the directory "graphs/paper-graphs/distance-based_10_13_14_35"
        :return: list of gxl-filenames
        """
        r_file_names = []
        # iterate over every file in the folder
        for file in os.listdir(folder):
            if file.endswith(".gxl"): #check if its a gxl file
                r_file_names.append(file) #add the filename to the list
        #return the list of gxl filenames
        return r_file_names

    def get_data_list(self, folder, k=0):
        """
        this function creates lists of data objects separated into train val and test
        :param k: k can take values from 0 to 3 and defines which datasplit in the 4-fold cross validation is used
        :return: lists of data objects for train val and test
        """

        data_split = self.split(k) #get dictionary that tells how to split the data into train val and test

        # initialize list
        train_data_list = []
        test_data_list = []
        val_data_list = []

        pattern = "_"
        a=0
        # iterate over every filename
        for file in self.raw_file_names(folder):
            # create the data object representing the graph stored in the file
            try:
                data = self.get_graph(folder, file)
            except:
                print(file, "could not be loaded")
                continue
            img = re.split(pattern, file)[0] # get the image from which this graph was sampled
            split = data_split[img][0]  # determine whether this graph belongs to train val or test

            # add the data object for the graph to the corresponding list
            if split == "test":
                test_data_list.append(data)
            elif split == "val":
                val_data_list.append(data)
            elif split == "train":
                train_data_list.append(data)

        return train_data_list, val_data_list, test_data_list


# class GraphDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(GraphDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         r_file_names = []
#         try:
#             for filename in os.listdir("./graph/paper-graphs/distance-based_10_13_14_35/"):
#                 if filename.endswith(".gxl"):
#                     r_file_names.append(filename)
#         except:
#             print("you have to save the raw gxl files in the following folder:")
#             print("graph/paper-graphs/distance-based_10_13_14_35/")
#
#         return r_file_names
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def download(self):
#         pass
#         # Download to `self.raw_dir`.
#
#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = []
#         for filename in self.raw_file_names():
#             data = get_graph(filename)
#             data_list.append(data)
#
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    folder = "graphs/paper-graphs/distance-based_10_13_14_35/"
    # folder = "graphs/base-dataset/"
    filename = "img0_0_normal.gxl"

    raw_data= DataConstructor()
    train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder)
    print(len(train_data_list))
    print(train_data_list[0])
    print("feature vec:",train_data_list[0].x[0])
    print(raw_data.get_graph(folder, filename))

    # Dataloader
    train_data = DataLoader(train_data_list, batch_size=32)
    for batch in train_data:
        print(batch.num_graphs)
        print(batch)