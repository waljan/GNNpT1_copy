#import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import re
import networkx as nx


def basic_info(filename):
    """
    this function just prints out some basic information about the graph

    :param: filename: gxl file name that stores a graph
    :return: all the node features and edges of the graph
    """

    tree = ET.ElementTree(file="graphs/paper-graphs/distance-based_10_13_14_35/" + filename)
    root = tree.getroot()

    # iterate over the nodes in the graph
    for node in root.iter("node"):

        # print the node label
        print ("Node:", node.attrib)

        # iterate over the features of the node
        for feature in node:
            # print the name of the feature
            print(feature.attrib, end=": ")
            # print the value of the feature
            for value in feature:
                print(value.text)
        print("---------------------------------")

    for edge in root.iter("edge"):
        print("Edge:", (edge.attrib))


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

    # # iterate over the nodes in the graph
    # for node in root.iter("node"):
    #     # iterate over the features of the node
    #     for feature in node:
    #         # print the name of the feature
    #         print(feature.attrib, end=": ")
    #         # print the value of the feature
    #         for value in feature:
    #             print(value.text)
    #     print("---------------------------------")



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


def main(filename):
    basic_info(filename)
    A = (adj_from_gxl(filename))
    dic = (get_node_features(filename))
    print (A)
    for key, value in dic.items():
        print ("Node" + key + ":", value)
        break


if __name__ == "__main__":

    filename = "img0_0_normal.gxl"
    main(filename)