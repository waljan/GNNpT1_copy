import xml.etree.ElementTree as ET
import numpy as np

# conda activate PyG373Msc

def basic_info(filename):
    """
    this function just prints out some basic information about the graph

    :param: filename: gxl file name that stores a graph
    :return: all the node features and edges of the graph
    """

    tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
    root = tree.getroot()

    # iterate over the nodes in the graph
    num_n = 0
    for node in root.iter("node"):
        num_n += 1
        # print the node label
        print ("Node:", node.attrib)

        # iterate over the features of the node
        num_f_perNode = 0
        for feature in node:
            num_f_perNode += 1
            # print the name of the feature
            print(feature.attrib, end=": ")
            # print the value of the feature
            for value in feature:
                print(value.text)
        print("---------------------------------")

    num_e = 0
    for edge in root.iter("edge"):
        num_e += 1
        print("Edge:", (edge.attrib))
    print("Graph", filename + ":\tnumber of nodes:", num_n, "\tnumber of node features:", num_f_perNode, "\tnumber of edges:", num_e )


def get_node_features(filename):
    """
    this function generates a dictionary containing all node features

    :param filename: gxl file that stores a graph
    :return: dictionary that has the node id as key (e.g. _12) and a list of features as value (stored as float)
    each node has a feature vector of length 6: attr_10, attr_13, attr_14, attr_35, x, y
    """

    # initialize tree
    tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
    root = tree.getroot()

    # initialize the dictionary
    graph_attributes = {}
    # iterate over every node
    for node in root.iter("node"):
        feature_vec = [float(value.text) for feature in node for value in feature]
        graph_attributes[node.attrib["id"]] = feature_vec

    return (graph_attributes)

def get_edges(filename):
    """
    This functions returns a list of lists containing the edges
    :param filename: gxl file that stores a graph
    :return: list of lists, each internal list contains the starting and endpoints of an edge
    """
    # initialize tree
    tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
    root = tree.getroot()

    # get the start and end points of every edge and store them in a list of lists
    start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root.iter("edge")]
    end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root.iter("edge")]
    edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

    return (edge_list)

def adj_from_gxl(filename):
    """
    this function takes a gxl file and returns adjacency matrix

    :param: filenam: gxl file that stores a graph
    :return: adjacency matrix of the graph
    """

    # read in the file by initializing the tree
    tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
    root = tree.getroot()

    # get the number of nodes in the graph
    i=0
    for node in root.iter("node"):
        i+=1

    # get the starting and end points of every edge
    start_points = [int("".join(filter(str.isdigit, edge.attrib["_from"]))) for edge in root.iter("edge")]
    end_points = [int("".join(filter(str.isdigit, edge.attrib["_to"]))) for edge in root.iter("edge")]

    #create the adjacency matrix
    A = np.zeros((i,i))
    for node in range(len(start_points)):
        A[start_points[node], end_points[node]]=1

    return (A)


import torch
from torch_geometric.data import Data


def get_graph(filename):
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

    if "normal" in filename:
        graph_class = 0
    elif "abnormal" in filename:
        graph_class = 1
    else:
        print("the filename has not the correct format")
    y = torch.tensor([graph_class], dtype=torch.float)

    # construct the graph
    graph = Data(x=x, edge_index=edge_index.t().contiguous())


    return (graph)


def main(filename):
    basic_info(filename)
    A = (adj_from_gxl(filename))
    dic = (get_node_features(filename))
    print (A)
    for key, value in dic.items():
        print ("Node" + key + ":", value)
        break
    g = get_graph(filename)
    print(g)


if __name__ == "__main__":
    filename = "img0_0_normal.gxl"
    main(filename)