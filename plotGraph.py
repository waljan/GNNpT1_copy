import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import read_gxl as readgxl


def plot_Graph(folder, filename):
    """
    this function takes a gxl file and a image and plots the cell graph on top of the image
    :param filename: gxl file that stores the graph
    :return: plots the graph on top of the image
    """
    #load the image
    p = filename.split("_")
    img_path = "pT1_dataset/dataset/" + p[0] + "/" + filename.replace(".gxl","") + "/" + filename.replace(".gxl","-image.jpg")
    # img=Image.open("dataset/img0/img0_0_normal/img0_0_normal-image.jpg")
    img=Image.open(img_path)

    img = np.asarray(img)
    # get the attributes of every node
    attr_dic =  readgxl.get_node_features(folder, filename)

    # get the coordinates of every node
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        coordinates = [[attr_dic[node][4], attr_dic[node][5]] for node in attr_dic]
    elif folder == "pT1_dataset/graphs/base-dataset/":
        coordinates = [[attr_dic[node][33], attr_dic[node][34]] for node in attr_dic]
    # scale the node coordinates to the image size
    for i in range(len(coordinates)):
        coordinates[i][0]*=(img.shape[1]-1)
        coordinates[i][1]*=(img.shape[0]-1)
    # get the edges of the graph
    edge_list = readgxl.get_edges(filename)

    # initialize the graph
    G = nx.Graph()
    G.add_edges_from(edge_list)


    nx.draw(G, coordinates, with_labels=False, node_shape=".", node_color="orangered", node_size=100, width=2, edge_color="orange" )

    # plt.savefig("../../images/" + filename.replace(".gxl","-Graph.png"), format="PNG")
    plt.show()

    # plt.imshow(img)
    # # plt.title(filename)
    # plt.axis("off")
    # plt.show()


    #plot the graph on top of the image
    plt.figure(1)
    plt.imshow(img)
    # plt.title(filename.replace(".gxl",""))
    nx.draw(G, coordinates, with_labels=False, node_shape=".", node_color="orangered", node_size=100, width=2, edge_color="orange" )
    plt.show()


if __name__ == "__main__":
    folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "pT1_dataset/graphs/base-dataset/"

    filename = "img0_0_normal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_1_normal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_2_normal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_3_normal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_12_abnormal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_13_abnormal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_14_abnormal.gxl"
    plot_Graph(folder, filename)
    filename = "img0_15_abnormal.gxl"
    plot_Graph(folder, filename)

