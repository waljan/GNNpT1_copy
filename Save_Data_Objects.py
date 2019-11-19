#!/usr/bin/python

import pickle

# own Modules
from Dataset_construction import DataConstructor
from Data_Augmentation import augment as aug

def save_obj(folder, augment):
    """
    saves training, validation, test and augmented training data lists for all 4 cross validation runs to a file
    :param folder: determeines the folder which contains the raw gxl files
    :param augment: integer that determines by which factor the dataset should be augmented, augment==0 means no augmentation

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
            train_data_list_aug = aug(train_data_list, n=augment)
            all_lists[3].append(train_data_list_aug)
        all_lists[0].append(train_data_list)
        all_lists[1].append(val_data_list)
        all_lists[2].append(test_data_list)

    # save the data lists
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        print("save datalists to Paper_Data_Lists.pickle")
        pickle_out = open("Paper_Data_Lists.pickle", "wb")
        pickle.dump(all_lists, pickle_out)
        pickle_out.close()
        print("done")
    if folder == "pT1_dataset/graphs/base-dataset/":
        print("save datalists to Base_Data_Lists.pickle")
        pickle_out = open("Base_Data_Lists.pickle", "wb")
        pickle.dump(all_lists, pickle_out)
        pickle_out.close()
        print("done")

def load_obj(folder):
    """
    Function that loads the data lists from the pickle file
    :param folder: determines whether the data lists from the paper-dataset or from the base dataset are loaded
    :return: returns the data lists

    all_lists contains 4 internal lists (train, val, test, train_augmented) of length 4 (4 fold cross validation)
    The first graph in the val set in run 2 of the cross validation is accessed by all_lists[1][2][0]

    all_lists[set][CV-run][graph]
    """
    if folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        pickle_in = open("Paper_Data_Lists.pickle", "rb")
        all_lists = pickle.load(pickle_in)
        pickle_in.close()
        return all_lists

    if folder == "pT1_dataset/graphs/base-dataset/":
        pickle_in = open("Base_Data_Lists.pickle", "rb")
        all_lists = pickle.load(pickle_in)
        pickle_in.close()
        return all_lists

if __name__ == "__main__":

    # choose folder
    ################################
    # folder = "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/"
    folder = "pT1_dataset/graphs/base-dataset/"

    # save data lists to pickle file
    ################################
    # augment=20
    # save_obj(folder, augment)


    # load data list from pickle file
    ################################
    all_lists = load_obj(folder)
    print(type(all_lists))
    print(all_lists[0][2][3])
    print(len(all_lists))
    print(len(all_lists[0]))
    print(len(all_lists[0][0]))
    print(len(all_lists[3]))
    print(len(all_lists[3][0]))