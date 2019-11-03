import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import os
import csv
import re

class DataConstructor():
    def __init__(self):
        pass
    def get_graph(self, filename):
        """
        this function takes a gxl-file as input and creates a graph as it is used in pytorch Geometric
        :param filename: gxl file that stores a graph
        :return python object modeling a single graph with various attributes
        """
        # initialize tree
        tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
        root = tree.getroot()

        # get the start and end points of every edge and store them in a list of lists
        start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root.iter("edge")]
        end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root.iter("edge")]
        edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        # create a tensor needed to construct the graph
        edge_index = torch.tensor(edge_list, dtype=torch.long)

        # initialize the list
        all_node_features = []
        # iterate over every node
        for node in root.iter("node"):
            # get the feature vector of every node
            feature_vec = [float(value.text) for feature in node for value in feature]
            # and append it to the list
            all_node_features.append(feature_vec)

        # create a tensor needed to construct the graph
        x = torch.tensor(all_node_features, dtype=torch.float)

        # add graph label
        if "abnormal" in filename:
            graph_class = 1
        elif "normal" in filename:
            graph_class = 0
        else:
            print("the filename has not the correct format")
        y = torch.tensor([graph_class], dtype=torch.float)

        # construct the graph
        graph = Data(x=x, y=y, edge_index=edge_index.t().contiguous())


        return (graph)


    def split(self, k):

        with open("dataset_split.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            dic = {}
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    dic[row[0]] = [row[k+1]]
        return dic


    # get a list of all filenames
    def raw_file_names(self):
        r_file_names = []
        for file in os.listdir("graphs/paper-graphs/distance-based_10_13_14_35"):
            if file.endswith(".gxl"):
                r_file_names.append(file)
        return r_file_names

    def get_data_list(self, k=0):
        # create a list of data objcts
        data_split = self.split(k)
        train_data_list = []
        test_data_list = []
        val_data_list = []

        pattern = "_"

        for filename in self.raw_file_names():
            data = self.get_graph(filename)
            img = re.split(pattern, filename)[0]
            split = data_split[img][0]
            #     print(split)
            if split == "test":
                test_data_list.append(data)
            elif split == "val":
                val_data_list.append(data)
            elif split == "train":
                train_data_list.append(data)

        return train_data_list, val_data_list, test_data_list
#


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
    filename = "img0_0_normal.gxl"
    raw_data= DataConstructor()
    train_data_list, val_data_list, test_data_list = raw_data.get_data_list()
    print(len(train_data_list))
    print(train_data_list[0])
    print(raw_data.get_graph(filename))

    # Dataloader
    train_data = DataLoader(train_data_list, batch_size=32)
    for batch in train_data:
        print(batch.num_graphs)
