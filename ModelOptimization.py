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
        """
        this functions takes a given set of hyperparameters and uses them to treain and validate a model.
        it returns the measure that we want to be minimized (in this case the negative accuracy)

        The objective function returns a dictionary.
        The fmin function looks for some special key-value pairs in the return value of the objective function
        which will be then passed to the optimization algorithm
        :param params: set of hyperparameters
        :return: dictionary containing the score and STATUS_OK
        """
        val_acc = []
        for it in range(3):                                     # train and validate the model 3 times to get an average accuracy
            print("run:", it)
            res, bool = train_and_val(**params, opt = True)     # train and validate the model using the given set of hyperparamters (**params)
            if bool:                                            # if the train_and_val function was stopped early, return the obtained score and go to the next param combination
                val_acc.append(res)
                score = mean(val_acc)
                return {"loss": -score, "status": STATUS_OK}    # stop the loop by returning the negative average accuracy obtained so far
            else:                                               # if the train_and_val function was completed dont stop the loop
                val_acc.append(res)

        score = mean(val_acc)                                   # compute the average accuracy from the 3 train and validation runs
        return {"loss": -score, "status": STATUS_OK}            # return the dictionary as it is needed by the fmin function
        # validation accuracy is used as a measure of performance. Because fmin tries to minimize the returned value the negative accuracy is returned

    trials = Trials()
    best_param = fmin(                  # fmin returns a dictionary with the best parameters
        fn = objective_function,        # fn is the function that is to be minimize
        space = search_space,           # searchspace for the parameters
        algo = tpe.suggest,             # Search algorithm: Tree of Parzen estimators
        max_evals = 25,                # number of parameter combinations that should be evalutated
        trials = trials,                # by passing a trials object we can inspect all the return values that were calculated during the experiment
        rstate=np.random.RandomState(111))      #
    print("done")

    loss = [x["result"]["loss"] for x in trials.trials]     # trials.trials is a list of dictionaries representing everything about the search
                                                            # loss is a list of all scores (negative accuracies) that were obtained during the experiment
    best_param_values = [x for x in best_param.values()]    # best_param is a dictionary with the parameter name as key and the best value as value
                                                            # best_param.values() output the best parameter values
                                                            # --> best_param_values is a list of the parametervalues that performed best

    print("train model with best parameters")
    print("best_param_values:", best_param_values)


    # some parameter values need to be transformed into a suitable format in order to feed them into the train_and_val function
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
    # train and validate the model using the best parameter values
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
        opt = False
    )

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss) * -1)        # min(loss) * -1 is the accuracy obtained with the best combination of parameter values
    print("Best parameters: ", best_param)                  # best_param is the dictionary containing the parameters as key and the best values as value
    print("Time elapsed: ", time() - start)
    print("Parameter combinations evaluated: ", 150)

    return trials       # the trials object contains all the return values calculated during the experiment


if __name__ == "__main__":
    # search space
    search_space = {
        # "batch_size": hp.choice("batch_size", np.arange(32, 65, 32, dtype=int)),
        "batch_size": pyll.scope.int(hp.quniform("batch_size", 32, 65, 32)),        # hp.quniform(label, low, high, q) returns a value like round(uniform(low, high)/q)*q   The used values will generate output of either 32.0 or 64.0
        "num_epochs": pyll.scope.int(hp.quniform("num_epochs", 5, 31, 5)),          # pyll.scope.int() converts the float output of hp.quiniform into an integer
        "num_layers": pyll.scope.int(hp.quniform("num_layers", 2, 6,1)),
        "num_input_features": hp.choice("num_input_features", [33]),                # hp.choice(label, obptions) options is a list from which one element will be chosen
        "hidden": pyll.scope.int(hp.quniform("hidden", 33, 133, 33)),
        "device": hp.choice("device", ["cuda"]),
        "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),                     # hp.loguniform(label, low, high) returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed
        "step_size": pyll.scope.int(hp.quniform("step_size", 2,11, 2)),
        "lr_decay": hp.uniform("lr_decay", 0.5, 1),                                 # hp.uniform(label, low, high) return a value uniformly between low and high
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