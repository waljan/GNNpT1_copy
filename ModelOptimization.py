#!/usr/bin/python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
import numpy as np
from statistics import mean
from time import time, strftime
import csv
# own modules
from main import train, evaluate, train_and_val, train_and_val_1Fold



def hyperopt(search_space, f, m, folder, augment, in_features, runs, iterations, device):
    """

    :param search_space: dictionary containing all the hyperparameters and the range that should be searched
    :param f: which fold of the 4 fold cross validation (integer from 0 to 3)
    :param m: which model (e.g. "GraphSAGE")
    :param folder: string: folder that contains the graphs
    :param augment: boolean: if False no data augmentation is performed
    :param in_features: number of input features
    :param runs: how often the model is trained and validated for a given set of hyperparameters
    :param iterations: how many different sets of hyperparameters are testes
    :param device: determines on which device the experiment is run ("cpu" or "cuda")
    :return:
    """
    start = time()
    print("fold: " + str(f) + " ####################################")
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
        for it in range(runs):                                     # train and validate the model 3 times to get an average accuracy
            print("run: "+ str(it) + " -------------------")
            # res, bool, _, _, _, _= train_and_val(**params, opt = True)     # train and validate the model using the given set of hyperparamters (**params)
            res, bool, _, _, _, _, _ = train_and_val_1Fold(**params, fold=f, m=m, opt=True, folder=folder, augment=augment, batch_size=32, num_input_features = in_features, device=device)
            if bool:                                            # if the train_and_val function was stopped early, return the obtained score and go to the next param combination
                val_acc.append(res)
                for k in range(runs-it-1):
                    val_acc.append(0.5)
                score = mean(val_acc)
                print("Model: " + str(m) + "   Dataset: " + str(folder) + "   runs evalutated: " + str(it+1))
                print("Parameters: " + str(params))
                print("score: " + str(score))
                print("############################################")
                return {"loss": -score, "status": STATUS_OK}    # stop the loop by returning the negative average accuracy obtained so far
            else:                                               # if the train_and_val function was completed dont stop the loop
                val_acc.append(res)

        score = mean(val_acc)                                   # compute the average accuracy from the 3 train and validation runs
        print("Model: " + str(m) +  "   Dataset: " + str(folder) + "   runs evalutated: " + str(it+1))
        print("Parameters: " + str(params))
        print("score: " + str(score))
        print("############################################")
        return {"loss": -score, "status": STATUS_OK}            # return the dictionary as it is needed by the fmin function
        # validation accuracy is used as a measure of performance. Because fmin tries to minimize the returned value the negative accuracy is returned

    trials = Trials()
    best_param = fmin(                  # fmin returns a dictionary with the best parameters
        fn = objective_function,        # fn is the function that is to be minimize
        space = search_space,           # searchspace for the parameters
        algo = tpe.suggest,             # Search algorithm: Tree of Parzen estimators
        max_evals = iterations,                # number of parameter combinations that should be evalutated
        trials = trials#,                # by passing a trials object we can inspect all the return values that were calculated during the experiment
        #rstate=np.random.RandomState(111)   
    )

    print("done")

    loss = [x["result"]["loss"] for x in trials.trials]     # trials.trials is a list of dictionaries representing everything about the search
                                                            # loss is a list of all scores (negative accuracies) that were obtained during the experiment
    best_param_values = [x for x in best_param.values()]    # best_param is a dictionary with the parameter name as key and the best value as value
                                                            # best_param.values() output the best parameter values
                                                            # --> best_param_values is a list of the parametervalues that performed best
    #
    # print("train model with best parameters")
    # print("best_param_values:", best_param_values)

    # # train_and_val(**best_param_values)
    # # train and validate the model using the best parameter values
    # train_and_val_1Fold(
    #     int(best_param["batch_size"]),
    #     int(best_param["num_epochs"]),
    #     int(best_param["num_layers"]),
    #     num_input_features,
    #     int(best_param["hidden"]),
    #     device,
    #     best_param["lr"],
    #     int(best_param["step_size"]),
    #     best_param["lr_decay"],
    #     m,
    #     folder,
    #     augment,
    #     fold=f,
    #     opt = False
    # )

    print("")
    print("##### Results " + m + "  fold: " + str(f) + "  folder:" + folder)
    print("Score best parameters: ", min(loss) * -1)        # min(loss) * -1 is the accuracy obtained with the best combination of parameter values
    print("Best parameters: ", best_param)                  # best_param is the dictionary containing the parameters as key and the best values as value
    print("Time elapsed: ", time() - start)
    print("Parameter combinations evaluated: ", iterations)
    print("############################################")
    print("############################################")
    print("############################################")

    # write final result to a file
    if folder == "pT1_dataset/graphs/base-dataset/":
        direc = "base/"
    elif folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        direc = "paper/"
    # path = "./out/" + direc + m + "/" + m + "-fold" + str(f) + "-r10-it100-" + strftime("%Y%m%d-%H%M%S") + ".csv"
    path = "./Hyperparameters/" + direc + m + "/" + m + "-fold" + str(f) + "-r10-it100.csv"
    with open(path, "w") as file:
        fieldnames = ["dataset", "model","fold", "num_evals", "num_runs_per_eval", "hidden", "lr", "lr_decay", "num_epochs", "num_layers", "step_size"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"dataset" : folder, "model" : m,"fold":f, "num_evals":iterations, "num_runs_per_eval":runs,
                            "hidden": best_param["hidden"], "lr" : best_param["lr"], "lr_decay" : best_param["lr_decay"],
                            "num_epochs" : best_param["num_epochs"], "num_layers" : best_param["num_layers"],
                            "step_size" : best_param["step_size"]})


    # return trials       # the trials object contains all the return values calculated during the experiment


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=41)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device",  type=str, default="cuda")
    args = parser.parse_args()

    if args.folder == "pT1_dataset/graphs/base-dataset/":
        in_features = 33
        low = 33
        high = 133
        step = 33
    elif args.folder == "pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/":
        in_features = 4
        low = 33
        high = 133
        step = 33


    # search space
    search_space = {
        "num_epochs": pyll.scope.int(hp.quniform("num_epochs", args.max_epochs-2, args.max_epochs, 10)),        # hp.quniform(label, low, high, q) returns a value like round(uniform(low, high)/q)*q  in the case of low=20 and high = 41 and q = 10 case: 20, 30,40
        "num_layers": pyll.scope.int(hp.quniform("num_layers", 2, 6,1)),            # pyll.scope.int() converts the float output of hp.quiniform into an integer
        "hidden": pyll.scope.int(hp.quniform("hidden", low, high, step)),
        "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.01)),                     # hp.loguniform(label, low, high) returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed
        "step_size": pyll.scope.int(hp.quniform("step_size", 3, 7, 3)),
        "lr_decay": hp.uniform("lr_decay", 0.5, 1),                                 # hp.uniform(label, low, high) return a value uniformly between low and high
    }                                                                               # hp.choice(label, obptions) options is a list from which one element will be chosen
    # folder= "pT1_dataset/graphs/base-dataset/"
    # augment=False


    hyperopt(search_space, f=args.fold, m=args.model, folder=args.folder, augment=args.augment, in_features = in_features, device=args.device, runs=args.runs, iterations=args.iterations)
    # print(type(results))


