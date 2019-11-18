#!/usr/bin/python
 # TODO: update requirements file
import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
import numpy as np
from statistics import mean
from time import time

# own modules
from model import GraphSAGE
from main import train, evaluate, train_and_val
from Save_Data_Objects import load_obj



def hyperopt(search_space):

    start = time()
    def objective_function(params):
        val_acc = []
        for it in range(3):
            print("iteration:", it)
            res, bool = train_and_val(**params)
            val_acc.append(res)
            if bool:
                score = mean(val_acc)
                return {"loss": -score, "status": STATUS_OK}
            else:
                val_acc.append(res)

        score = mean(val_acc)
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    best_param = fmin(                        # fmin returns a dictionary with the best parameters
        fn = objective_function,               # fn is the function that is to be minimize
        space = search_space,
        algo = tpe.suggest,             # Search algorithm: Tree of Parzen estimators
        max_evals = 4,
        trials = trials,
        rstate=np.random.RandomState(111))
    print("done")

    loss = [x["result"]["loss"] for x in trials.trials]

    best_param_values = [x for x in best_param.values()]

    print("train model with best parameters")
    print("best_param_values:", best_param_values)

    if best_param_values[0] == 0:
        augment = False

    if best_param_values[2] == 0:
        device = "cuda"
    if best_param_values[2] == 1:
        device = "cpu"

    if best_param_values[3] == 0:
        folder = "pT1_dataset/graphs/base-dataset/"

    if best_param_values[7] == 0:
        m = "GraphSAGE"

    if best_param_values[9] == 0:
        num_input_features = 33

    # train_and_val(**best_param_values)
    train_and_val(
        int(best_param["batch_size"]),
        int(best_param["num_epochs"]),
        int(best_param["num_layers"]),
        num_input_features,
        int(best_param["hidden"]),
        device,
        best_param["lr"],
        int(best_param["step_size"]),
        best_param["lr_decay"],
        m,
        folder,
        augment
    )

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss) * -1)
    print("Best parameters: ", best_param)
    print("Time elapsed: ", time() - start)
    print("Parameter combinations evaluated: ", 10)

    return trials


if __name__ == "__main__":
    # search space
    search_space = {
        # "batch_size": hp.choice("batch_size", np.arange(32, 65, 32, dtype=int)),
        "batch_size": pyll.scope.int(hp.quniform("batch_size", 32, 65, 32)),
        "num_epochs": pyll.scope.int(hp.quniform("num_epochs", 5, 31, 5)),
        "num_layers": pyll.scope.int(hp.quniform("num_layers", 2, 6,1)),
        "num_input_features": hp.choice("num_input_features", [33]),
        "hidden": pyll.scope.int(hp.quniform("hidden", 33, 133, 33)),
        "device": hp.choice("device", ["cuda"]),
        "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
        "step_size": pyll.scope.int(hp.quniform("step_size", 2,11, 2)),
        "lr_decay": hp.uniform("lr_decay", 0.5, 1),
        "m": hp.choice("m", ["GraphSAGE"]),
        "folder": hp.choice("folder", ["pT1_dataset/graphs/base-dataset/"]),
        "augment": hp.choice("augment", [0])
    }

    results = hyperopt(search_space)
    print(type(results))

# # data
# folder = "pT1_dataset/graphs/base-dataset/"
# all_lists = load_obj(folder)
# all_train_lists = all_lists[0]
# all_val_lists = all_lists[1]
#
# # the objective function takes a set of hyperparameters as input and outputs a score
# def objective_function(params, all_train_lists, all_val_lists):
#
#
#     batch_size = 32
#     device = "cuda"
#
#     for iter in range(10):
#     for k in range(4):      # 4 fold cross validation
#         train_data_list = all_train_lists[k]
#         val_data_list = all_val_lists[k]
#
#         model = GraphSAGE(
#             num_layers=params["num_layers"],
#             num_input_features=params["num_input_features"],
#             hidden=params["hidden"]
#         ).to(device)
#
#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=params["lr"],
#             weight_decay=0.001)  # define the optimizer, weight_decay corresponds to L2 regularization
#
#         scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["lr_decay"])  # learning rate decay
#
#         crit = torch.nn.MSELoss()
#
#         # initialize train loader
#         train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
#         # initialize val loader
#         val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)
#
#         for epoch in range(params["num_epochs"]):
#             # train the model
#             train(model, train_loader, optimizer, crit)
#             scheduler.step()
#
#             train_acc, _, _, loss = evaluate(model, train_loader, crit)  # compute the accuracy for the training data
#
#             val_acc, predictions, labels, val_loss = evaluate(model, val_loader,
#                                                               crit)  # compute the accuracy for the test data
#             val_accs[k].append(val_acc)
#             val_losses[k].append(val_loss)