#!/usr/bin/python

import pickle
import torch

# own Modules
from Dataset_construction import DataConstructor
from Data_Augmentation import augment as aug

def save_obj(folder, augment, sd=1):
    """
    saves training, validation, test and augmented training data lists for all 4 cross validation runs to a file

    :param folder: (str) determeines the folder which contains the raw gxl files
    :param augment: integer that determines by which factor the dataset should be augmented, augment==0 means no augmentation
    :param sd: (float or int) determines the standard deviation of the normal distribution used in the augmentation function

    The filename does contain information about folder, augment and sd.
    For sd, if it is a float, "." of the float will be replaced by "-" in the filename:
    If sd was a float such as 0.5, the filename will contain 0-5. If sd was an int such as 1, the filename will contain 1

    the list "all_lists" is saved to the file data_lists.pickle
    all_lists contains 4 internal lists (for train, val, test, train_augmented) of length 4 (4 fold cross validation)
    The first graph in the val set in run 2 of the cross validation is accessed by all_lists[1][2][0]

    all_lists[set][CV-run][graph]
    """
    print("load data")
    raw_data = DataConstructor()            # create instance of the class DataConstructor()

    all_lists=[[],[],[],[]]                 # initialize the list that will contain the four data lists
    print("create data lists")
    for k in range(4):                      # 4 iterations for the 4 fold cross validation
        print("k:", k)
        train_data_list, val_data_list, test_data_list = raw_data.get_data_list(folder, k=k)  # split the data into train val and test

        # augment data by adding/subtracting small random values from node features
        if augment > 1:
            train_data_list_aug = aug(train_data_list, n=augment, sd=sd)

            # normalize node features
            # TODO: how is the original data normalized? per split, or all included?
            ####
            first_it = True
            for graph in train_data_list_aug:       # iterate over every graph
                if first_it:
                    features = graph.x                     # get the tensor [num_nodes, num_features] containing all node features of the graph
                    first_it = False
                else:
                    features = torch.cat((features, graph.x))     # concatenate the feature tensors of all graphs
            f_avg = features.mean(dim=0, keepdim=True)     # compute the mean for every feature across all nodes of all graphs
            f_sd = features.std(dim=0, keepdim=True)       # compute the standard deviation for every feature across all nodes of all graphs

            for graph_idx in range(len(train_data_list_aug)):
                train_data_list_aug[graph_idx].x = (train_data_list_aug[graph_idx].x - f_avg)/f_sd      # standardize the features to have mean=0, std=0
            ####



            all_lists[3].append(train_data_list_aug)
        all_lists[0].append(train_data_list)
        all_lists[1].append(val_data_list)
        all_lists[2].append(test_data_list)

    # save the data lists
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        print("save datalists to Data_Lists/Paper_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle")
        pickle_out = open("Data_Lists/Paper_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle", "wb")
        pickle.dump(all_lists, pickle_out)
        pickle_out.close()
        print("done")
    if folder == "pT1_dataset/graphs/base-dataset/":
        print("save datalists to Data_Lists/Base_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle")
        pickle_out = open("Data_Lists/Base_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle", "wb")
        pickle.dump(all_lists, pickle_out)
        pickle_out.close()
        print("done")

def load_obj(folder, augment, sd):
    """
    Function that loads the data lists from the pickle file
    :param folder: (str) determines whether the base or paper graph dataset is used
    :param augment: (int) determines by which factor the dataset should be augmented, augment==0 means no augmentation
    :param sd: (float or int) determines the standard deviation of the normal distribution used in the augmentation function
    :return: returns the data lists

    all_lists contains 4 internal lists (train, val, test, train_augmented) of length 4 (4 fold cross validation)
    The first graph in the val set in run 2 of the cross validation is accessed by all_lists[1][2][0]

    all_lists[set][CV-run][graph]
    """
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        pickle_in = open("Data_Lists/Paper_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle", "rb")
        all_lists = pickle.load(pickle_in)
        pickle_in.close()
        return all_lists

    if folder == "pT1_dataset/graphs/base-dataset/":

        pickle_in = open("Data_Lists/Base_Data_Lists_sd" + str(sd).replace(".","-") + "_aug" + str(augment) + ".pickle", "rb")
        all_lists = pickle.load(pickle_in)
        pickle_in.close()
        return all_lists

if __name__ == "__main__":

    ## choose folder
    ################################
    # folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "pT1_dataset/graphs/base-dataset/"

    ## choose augment
    ###############################
    augment = 10

    # choose sd
    ##############################
    sd = 0.01
    # sd = 0.5

    # ## save data lists to pickle file
    # ###############################
    # save_obj(folder, augment, sd)


    # load data list from pickle file
    ################################
    all_lists = load_obj(folder, augment, sd)       # all_lists[set][CV-run][graph]
    print(type(all_lists))
    print(all_lists[0][2][3])
    print(len(all_lists))
    print(len(all_lists[0]))
    print(len(all_lists[0][0]))
    print(len(all_lists[3]))
    print(len(all_lists[3][0]))

    all_train_lists = all_lists[0]
    all_val_lists = all_lists[1]
    all_test_lists = all_lists[2]
    all_train_aug_lists = all_lists[3]


    import matplotlib.pyplot as plt
    from statistics import stdev, mean

    plt.rc("font", size=5)  # change font size

    for f_idx in range(all_train_aug_lists[0][0].num_node_features):
        f_vec = []

        for graph in all_train_aug_lists[0]:
            for nd_idx in range(graph.num_nodes):
                f_vec.append(graph.x[nd_idx][f_idx].item())

        plt.subplot(4, all_train_aug_lists[0][0].num_node_features//4+1, f_idx+1)
        plt.hist(f_vec, density=True)
        plt.title("f"+ str(f_idx) + "  std: " + str(stdev(f_vec))[:4] + "  mean: " + str(mean(f_vec))[:4])

    # plt.tight_layout()
    plt.show()