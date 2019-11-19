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
            res, bool = train_and_val(**params, opt = True)
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
        max_evals = 150,
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
        augment,
        opt = True
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


# 150 iterations for GraphSage on base-dataset
# Score best parameters:  0.9550437288809382
# Best parameters:  {'augment': 0, 'batch_size': 32.0, 'device': 0, 'folder': 0, 'hidden': 33.0, 'lr': 0.002000837078491707, 'lr_decay': 0.9054101608439347, 'm': 0, 'num_epochs': 30.0, 'num_input_features': 0, 'num_layers': 2.0, 'step_size': 10.0}
# Time elapsed:  3047.017801761627
# Parameter combinations evaluated:  10