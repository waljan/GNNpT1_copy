import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import read_gxl as readgxl


def plot_Graph(filename):
    """
    this function takes a gxl file and a image and plots the cell graph on top of the image
    :param filename: gxl file that stores the graph
    :return: plots the graph on top of the image
    """
    #load the image
    img=Image.open("dataset/img0/img0_0_normal/img0_0_normal-image.jpg") #TODO this path should automatically be chosen according to the input filename
    img = np.asarray(img)
    # get the attributes of every node
    attr_dic =  readgxl.get_node_features(filename)
    # get the coordinates of every node
    coordinates = [[attr_dic[node][4], attr_dic[node][5]] for node in attr_dic]
    # scale the node coordinates to the image size
    for i in range(len(coordinates)):
        coordinates[i][0]*=img.shape[1]
        coordinates[i][1]*=img.shape[0]
    # get the edges of the graph
    edge_list = readgxl.get_edges(filename)

    # initialize the graph
    G = nx.Graph()
    G.add_edges_from(edge_list)

    #plot the graph on top of the image
    plt.figure(1)
    plt.imshow(img)
    nx.draw_networkx(G, coordinates)
    plt.show()


if __name__ == "__main__":
    filename = "img0_0_normal.gxl"
    plot_Graph(filename)
