#!/usr/bin/python
from __future__ import absolute_import
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16_bn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import stdev, mean
import csv
import re



def plot_imgs():
    img_path = "pT1_dataset/dataset/img0/img0_11_normal/img0_11_normal-image.jpg"
    # mask_path = "pT1_dataset/dataset/img0/img0_11_normal/img0_11_normal-gt.png"
    img = Image.open(img_path)
    # mask = Image.open(mask_path)
    img = np.asarray(img)
    # mask = np.asarray(mask)
    # mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
    # img = np.where(mask, img, mask)
    plt.imshow(img)
    plt.show()

    img_path = "pT1_dataset/dataset/img1/img1_11_abnormal/img1_11_abnormal-image.jpg"
    # mask_path = "pT1_dataset/dataset/img1/img1_11_abnormal/img1_11_abnormal-gt.png"
    img = Image.open(img_path)
    # mask = Image.open(mask_path)
    img = np.asarray(img)
    # mask = np.asarray(mask)
    # mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
    # img = np.where(mask, img, mask)
    plt.imshow(img)
    plt.show()

def image_sizes():
    path = "pT1_dataset/dataset/"
    smallest_width = 10000
    smallest_hight = 10000
    for patient in os.listdir(path):
        if not patient.endswith(".csv"):
            for img_folder in os.listdir(path + patient):
                img = Image.open(path+patient+ "/" + img_folder + "/" + img_folder + "-image.jpg")
                img = np.asarray(img)
                if img.shape[0] < smallest_hight:
                    smallest_hight = img.shape[0]
                    pic_h = img_folder
                if img.shape[1] < smallest_width:
                    smallest_width = img.shape[1]
                    pic_w = img_folder
    print(smallest_hight, pic_h)
    print(smallest_width, pic_w)

        # for img in os.listdir(path + paient)
        # if not f.startswith("."):
        #
        #     if os.path.isfile(path + f) and k in f:
        #         with open(path + f, "r") as file:

def plot_all_images():
    """
    plots all images of the dataset
    :return:
    """
    path = "pT1_dataset/dataset/"
    counter = 1
    for patient in os.listdir(path):                        # iterate over every patient
        if not patient.endswith(".csv"):                    # only consider the img folders
            for img_folder in os.listdir(path + patient):   # iterate ofrer ever img folder

                img = Image.open(path+patient+ "/" + img_folder + "/" + img_folder + "-image.jpg")  # open the image (PIL)
                img = np.asarray(img)       # convert from PILformat to numpy array
                if counter <= 100:
                    plt.rc("font", size=5)      # determine font size
                    plt.subplot(10,10, counter)
                    plt.imshow(img)
                    if "abnormal" in img_folder:
                        plt.title("dysplastic")
                    else:
                        plt.title("normal")
                    plt.axis("off")
                    counter+=1
                else:
                    plt.show()
                    counter=1
                    plt.rc("font", size=5)
                    plt.subplot(10, 10, counter)
                    plt.imshow(img)
                    if "abnormal" in img_folder:
                        plt.title("dysplastic")
                    else:
                        plt.title("normal")
                    plt.axis("off")
                    counter+=1


############################################################################################

def get_opt_param(fold):
    """
    reads a csv file to  return the optimal parameters for a given data-split
    """

    k = "fold" + str(fold)
    path = "Hyperparameters/CNN/"
    for f in os.listdir(path):
        if not f.startswith("."):
            if os.path.isfile(path + f) and k in f and f.endswith("it100.csv"):
                with open(path + f, "r") as file:
                    hp = list(csv.reader(file))
                    hp=hp[1]
                    m = hp[0]
                    fold = int(hp[1])
                    opt_lr = float(hp[4])
                    opt_lr_decay = float(hp[5])
                    opt_num_epochs = int(float(hp[6]))
                    opt_step_size = int(float(hp[7]))
                    opt_weight_decay = float(hp[8])
    return m, fold, opt_lr, opt_lr_decay, opt_num_epochs, opt_step_size, opt_weight_decay

def train_val_test_split(fold):
    """
    :param fold: determines which data split is used
    :return: three (train, val, test) lists containing the IDs of the images,
                the ID is like he path to the image, ID looks like: img0/img0_0_normal/img0_0_normal-image.jpg
    """
    # open the csv file
    with open("pT1_dataset/dataset_split.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        dic = {}

        # iterate over every row of the csv file
        for row in csv_reader:
            if line_count == 0:  # ignore the header
                line_count += 1
            else:  # use a dictionary to save the information about how to split the data into train test and val
                dic[row[0]] = [row[fold + 1]]   # get a dictionary containing all the needed information to split the data

    path = "pT1_dataset/dataset/"
    train_IDs, val_IDs, test_IDs = [],[],[]
    for patient in os.listdir(path):        # iterate over the diretory (iterate over every patient)
        if not patient.endswith(".csv"):    # ignore the csv file in this folder
            if dic[patient][0]=="train":    # check if the patient belongs to train
                for img_folder in os.listdir(path + patient):   # iterate over all images from this patient
                    train_IDs.append(patient + "/" + img_folder + "/" + img_folder + "-image.jpg")  # append the ID
            if dic[patient][0]=="val":
                for img_folder in os.listdir(path + patient):
                    val_IDs.append(patient + "/" + img_folder + "/" + img_folder + "-image.jpg")
            if dic[patient][0] == "test":
                for img_folder in os.listdir(path + patient):
                    test_IDs.append(patient + "/" + img_folder + "/" + img_folder + "-image.jpg")
    return train_IDs, val_IDs, test_IDs

############################################################################################


class ImageDataset(Dataset):
    def __init__(self, list_IDs, transform):
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        img = Image.open("pT1_dataset/dataset/" + ID)

        # img = transforms.Resize([64,60])(img) #TODO resize images
        if "abnormal" in ID:
            label = [1,0]
            cls = 1
        else:
            label = [0,1]
            cls = 0

        label = torch.tensor(label, dtype=torch.float)

        img = self.transform(img)

        _, _, filename = ID.split("/")
        img_patch = [int(i) for i in re.findall(r"[0-9]+", filename)]
        img_patch.append(cls)
        name = torch.tensor([img_patch], dtype=torch.float)

        # print(img.shape)
        return img, label, name

class CNN(nn.Module):
    """
    feed forward conv net
    """
    def __init__(self, img_size):
        super(CNN, self).__init__()
        self.final_img_size = int(img_size/8)
        self.out_conv1= 16
        self.out_conv2 = 32
        self.out_conv3 = 64
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_conv1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=self.out_conv1, out_channels=self.out_conv2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=self.out_conv2, out_channels=self.out_conv3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
        )
        self.linear_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.final_img_size*self.final_img_size*self.out_conv3, out_features=self.final_img_size*self.final_img_size*16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.final_img_size*self.final_img_size*16, out_features=2),
            nn.Softmax()
        )

    def forward(self, input):
        output = self.cnn_layers(input)
        output_flat = output.reshape(-1, self.final_img_size*self.final_img_size*self.out_conv3)
        output = self.linear_layers(output_flat)
        return output

############################################################################################




def train_and_test_1Fold(fold, m, device):
    # get the hyperparameters
    _, _, lr, lr_decay, num_epochs, step_size, weight_decay = get_opt_param(fold)
    bool=True
    while bool:
        # train and validate the CNN, save the model weights that have highest accuracy on validation set
        val_res, bool, train_accs, val_accs, train_losses, val_losses = train_and_val_1Fold(fold=fold, m=m,
                                                                                                          num_epochs=num_epochs,
                                                                                                          lr=lr,
                                                                                                          lr_decay=lr_decay,
                                                                                                          weight_decay=weight_decay,
                                                                                                          device=device,
                                                                                                          step_size=step_size,
                                                                                                          testing=True)
        if bool:
            print("FOLD" + str(fold) + "rerun due to dead neurons")
    print("val acc:", val_res[2])

    img_size = 128
    train_IDs, val_IDs, test_IDs = train_val_test_split(fold)                           # split data

    val_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor()])
    test_data = ImageDataset(test_IDs, val_transform)                                   # get data objects
    # data loaders
    batchsize = 64
    test_loader = DataLoader(test_data, batchsize, shuffle=True)                        # create Dataloader



    model = torch.load("Parameters/CNN/" + m + "_fold"+str(fold) + ".pt")                     # load model
    crit = torch.nn.CrossEntropyLoss(reduction="sum")                                   # define loss function
    test_acc, test_loss, img_name, TP_TN_FP_FN = evaluate(model, test_loader, crit, device, testing=True)     # evaluation on test_set
    print("test_acc:", test_acc)
    # print(TP_TN_FP_FN)
    return test_acc, train_accs, val_accs, train_losses, val_losses, img_name, TP_TN_FP_FN

def test(runs, m,  device):
    """
    write train accs and val accs of all runs to one csv file per fold
    :param runs: number of times train, val and test is repeated
    :param device: "cuda" or "cpu"
    :return:
    """
    # m = "CNN"
    folder_short="CNN/"
    all_test_accs = []
    test_accs_per_fold = [[],[],[],[]]

    path_test = "out/" + folder_short + m + "_test_data.csv"

    for fold in range(4):
        print("Fold:", fold)

        # paths of csv files to which will be written to
        path_train_acc = "out/" + folder_short + m + "_train_acc_fold" + str(fold) + ".csv"
        path_train_loss = "out/" + folder_short + m + "_train_loss_fold" + str(fold) + ".csv"

        path_val_acc = "out/" + folder_short + m + "_val_acc_fold" + str(fold) + ".csv"
        path_val_loss = "out/" + folder_short + m + "_val_loss_fold" + str(fold) + ".csv"

        path_TP_TN_FP_FN = "out/" + folder_short + m + "_TP_TN_FP_FN_fold" + str(fold) + ".csv"

        # write train acc, train loss, val acc and val loss to separate csv files
        with open(path_train_acc, "w") as train_acc_file, \
                open(path_train_loss, "w") as train_loss_file, \
                open(path_val_acc, "w") as val_acc_file, \
                open(path_val_loss, "w") as val_loss_file, \
                open(path_TP_TN_FP_FN, "w") as cls_file:

            train_acc_writer = csv.writer(train_acc_file)
            train_loss_writer = csv.writer(train_loss_file)

            val_acc_writer = csv.writer(val_acc_file)
            val_loss_writer = csv.writer(val_loss_file)

            cls_writer = csv.writer(cls_file)

            for it in range(runs):
                test_acc, train_accs, val_accs, train_losses, val_losses, img_name, TP_TN_FP_FN = train_and_test_1Fold(fold=fold, m=m, device=device)

                all_test_accs.append(test_acc)
                test_accs_per_fold[fold].append(test_acc)

                train_acc_writer.writerow([i for i in train_accs])
                train_loss_writer.writerow([i for i in train_losses])

                val_acc_writer.writerow([i for i in val_accs])
                val_loss_writer.writerow([i for i in val_losses])

                final_TPTNFPFN = []
                groups = ["TP", "TN", "FP", "FN"]
                for group_idx in range(4):
                    for img in img_name[group_idx]:
                        img_str = str(img[0][0]) + "_" + str(img[0][1]) + "_" + str(img[0][2]) + "_" + groups[group_idx]
                        final_TPTNFPFN.append(img_str)

                cls_writer.writerow(final_TPTNFPFN)

                if it == runs-1:
                    avg = mean(test_accs_per_fold[fold])
                    sd = stdev(test_accs_per_fold[fold])
                    test_accs_per_fold[fold].append(avg)
                    test_accs_per_fold[fold].append(sd)
                    test_accs_per_fold[fold].reverse()
    # write test results (mean and sd) of every fold and total to a csv file
    with open(path_test, "w") as test_file:
        test_writer = csv.writer(test_file)
        test_writer.writerow([stdev(all_test_accs), mean(all_test_accs)])
        for fold in range(4):
            test_writer.writerow(test_accs_per_fold[fold])

    print("Results on Testset:")
    print("mean:", "\t",  mean(all_test_accs)*100)
    print("standard deviation:", "\t", stdev(all_test_accs)*100)



############################################################################################


def train_and_val_1Fold(fold, m, num_epochs, lr, lr_decay, step_size, weight_decay, device, plotting=False, testing=False):
    img_size = 128
    # img_size = 256
    train_IDs, val_IDs, test_IDs = train_val_test_split(fold)
    train_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.RandomRotation((0,360)),
                                    # transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor()])
    train_data = ImageDataset(train_IDs, train_transform)
    val_data = ImageDataset(val_IDs, val_transform)
    test_data = ImageDataset(test_IDs, val_transform)
    # print("train size: " + str(len(train_data)) + "   val size: " + str(len(val_data)))
    # data loaders
    batchsize = 64
    # batchsize = 16
    train_loader = DataLoader(train_data, batchsize, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batchsize, shuffle=True)

    train_accs = []  # will contain the training accuracy of every epoch
    val_accs = []  # will contain the validation accuracy of every epoch

    train_losses = []  # will contain the training loss of every epoch
    val_losses = []  # will contain the validation loss of every epoch

    # initialize model
    print("initialize", m)
    if m == "CNN":
        model = CNN(img_size).to(device)
    elif m == "VGG16_bn":
        model = vgg16_bn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # define the optimizer, weight_decay corresponds to L2 regularization
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)  # learning rate decay
    crit = torch.nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(num_epochs):
        if epoch == 0: # get train and val accs before training
            train_acc, train_loss, _, _ = evaluate(model, train_loader, crit, device)
            val_acc, val_loss, img_name, TP_TN_FP_FN = evaluate(model, val_loader, crit, device)

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            running_val_acc = np.array([0,0,val_acc])
            val_res = np.copy(running_val_acc)

            if testing:
                torch.save(model, "Parameters/CNN/" + m + "_fold"+str(fold) + ".pt")
        # train the model
        train(model, train_loader, optimizer, crit, device)
        scheduler.step()

        # evalutate the model
        train_acc, train_loss, _, _= evaluate(model, train_loader, crit, device)
        val_acc, val_loss, img_name, TP_TN_FP_FN = evaluate(model, val_loader, crit, device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        running_val_acc[0] = running_val_acc[1]
        running_val_acc[1] = running_val_acc[2]
        running_val_acc[2] = val_acc

        if np.mean(running_val_acc) > np.mean(val_res) and not testing:
            val_res = np.copy(running_val_acc)
        if running_val_acc[2] > val_res[2] and testing:
            val_res = np.copy(running_val_acc)
            torch.save(model,"Parameters/CNN/" + m + "_fold"+str(fold) + ".pt")
        if plotting:
            for param_group in optimizer.param_groups:
                print("Epoch: {:03d}, lr: {:.5f}, train_loss: {:.5f}, val_loss: {:.5f}, train_acc: {:.5f}, val_acc: {:.5f}".format(epoch, param_group["lr"], train_loss, val_loss, train_acc, val_acc))

    if stdev(train_losses[-20:]) < 0.05 and mean(train_accs[-20:]) < 0.55:
        boolean = True
        # print("Oops")
    else:
        boolean = False

    # # plot learning curves
    if plotting:
        x = np.arange(0,len(train_accs))
        plt.subplot(2,1,1)
        plt.plot(x, train_accs, color="r")
        plt.ylim(0.5, 1)
        plt.plot(x, val_accs, color="g")
        plt.subplot(2,1,2)
        plt.plot(x, train_losses, color="r")
        plt.plot(x, val_losses, color="g")
        plt.show()
    return(val_res, boolean, np.asarray(train_accs), np.asarray(val_accs), np.asarray(train_losses), np.asarray(val_losses))


def train(model, train_loader, optimizer, crit, device):
    model.train()
    for data, label, name in train_loader:
        data = data.to(device) # transfer the data to the device
        label = label.to(device)  # transfer the labels to the device
        optimizer.zero_grad() # set the gradient to 0
        output = model(data) # pass the data through the model



        loss = crit(output, torch.max(label,1)[1].long()) # compute the loss between output and label
        loss.backward() # compute the gradient
        optimizer.step()

def evaluate(model, val_loader, crit, device, testing=False):
    model.eval()
    loss_all =0
    img_count=0
    batch_count =0
    correct_pred = 0
    img_name = [[],[], [], []]
    TP_TN_FP_FN = np.zeros((4))
    with torch.no_grad(): # gradients don't need to be calculated in evaluation

        # pass data through the model and get label and prediction
        for data, labelT, name in val_loader: # iterate over every batch in validation training set
            data = data.to(device) # trainsfer data to device
            predT = model(data)#.detach().cpu().numpy()   # pass the data through the model and store the predictions in a numpy array
            pred = predT.detach().cpu().numpy()
            label=labelT.detach().cpu().numpy()
            predicted_classes = (pred == pred.max(axis=1)[:,None]).astype(int)
            # if testing:
            #     c=0
            #     for i in data:
            #         plt.imshow(np.transpose(i.detach().cpu().numpy(), (1,2,0)))
            #         plt.title("predicted:" +str(predicted_classes[c,0]) + "    class:" +str(label[c,0]))
            #         plt.show()
            #         c+=1
            correct_pred += np.sum(predicted_classes[:, 0] == labelT[:, 0].numpy())

            loss = crit(predT, torch.max(labelT, 1)[1].long().to(device)) # compute the loss between output and label
            loss_all += loss.item()

            img_count += len(data)
            batch_count +=1



            # count the false negatives and false positives
            false_idx = np.argwhere(predicted_classes[:,0]!=label[:,0]).reshape(-1)
            truth = label[false_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[3] +=1
                    img_name[3].append(name[false_idx][c].tolist())
                if t[0]  == 0:
                    TP_TN_FP_FN[2] +=1
                    img_name[2].append(name[false_idx][c].tolist())
                c+=1

            true_idx = np.argwhere(predicted_classes[:,0]==label[:,0]).reshape(-1)
            truth = label[true_idx,:]
            c=0
            for t in truth:
                if t[0] == 1:
                    TP_TN_FP_FN[0] +=1
                    img_name[0].append(name[true_idx][c].tolist())
                if t[0] == 0:
                    TP_TN_FP_FN[1] += 1
                    img_name[1].append(name[true_idx][c].tolist())
                c+=1



    avg_acc = correct_pred / img_count
    avg_loss = loss_all/img_count
    return avg_acc, avg_loss, img_name, TP_TN_FP_FN



if __name__ == "__main__":
    # plot_imgs()
    # image_sizes()
    # split_images(0)


    # train_list_ID0, _, _ = train_val_test_split(0)
    # print("num_train_samples:", len(train_list_ID0))
    # transform = transforms.Compose([transforms.Resize((128, 128)),
    #                                transforms.ToTensor()])
    # train_split0 = ImageDataset(train_list_ID0, transform)
    #
    #
    # train_loader = DataLoader(train_split0, batch_size=32)
    # for batch, labels in train_loader:
    #     print("batchsize:", len(batch))
    #     for idx, img in enumerate(batch):
    #         plt.subplot(8,4, idx+1)
    #         print(img.size())
    #         tr = transforms.ToPILImage()
    #         image = tr(img)
    #         print(image.size)
    #         image = np.asarray(image)
    #         plt.imshow(np.asarray(image))
    #         if labels[idx].item() == 0:
    #             ttl = "normal"
    #         else:
    #             ttl = "dysplastic"
    #         plt.title(ttl)
    #     plt.show()


    # plot_all_images()
    # train_and_val_1Fold(fold=0, m="VGG16_bn", num_epochs=30, lr=0.01, lr_decay=0.8, step_size=3, weight_decay=0.01, device="cuda", plotting=True)


    test(runs=100,device="cuda", m="VGG16_bn")