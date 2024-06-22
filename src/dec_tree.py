
from utils import count_frauds, augment_train_data

from dataclasses import dataclass, replace
from random import randint 
import pickle
import pandas as pd
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot

@dataclass
class ModelArgs:
    model_out_file : str
    criterion : str = "gini"
    splitter : str = "best"
    max_depth : int = None
    max_features : int = None
    random_state : int = None
    ccp_alpha : float = 0.0
    min_impurity_decrease : float = 0.0
    test_size : int = 0
    train : bool = False

# Load the data globally as it is required by training and testing and will not be needed to be loaded during each brute-force search step.
train_data = pd.read_csv("data/transactions.csv")

X = train_data.drop(columns = "Class")
Y = train_data["Class"]

def search(model_out_file):
    highest = 0.0
    hyp_params = None
    model = None

    r = randint(0, 200000)

    with open("search_results.csv", "w+") as file:
        file.write(f"criterion,splitter,max_depth,max_features,random_state,ccp_alpha,min_impurity_decrease,test_size,train_acc,test_acc\n")

        for c in ["gini", "entropy", "log_loss"]:
            for s in ["best", "random"]:
                for a in [0.0, 0.00001]:
                    for mi in [0.0,  0.00001]:
                        for m in [None, 15]:
                            for mf in [None, 15]:
                                for ts in [0.3]:
                                    p = ModelArgs("", c, s, m, mf, r, a, mi, ts, True)
                                    (curr_model, train_score, test_score) = main(p)
                                    file.write(f"{c},{s},{m},{mf},{r},{a},{mi},{ts},{train_score},{test_score}\n")
                                    if (test_score > highest):
                                        print(f"New Highscore: {test_score}, {train_score} using {p}")
                                        highest = test_score
                                        highest_train = train_score
                                        hyp_params = replace(p)
                                        model = curr_model

    print(f"Highest Score {highest}, {highest_train} using {hyp_params}")

    with open(model_out_file, "wb+") as file:
        pickle.dump(model, file)

def main(model_args: ModelArgs, print_metrics = False): 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=model_args.test_size, random_state=model_args.random_state)

# Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_args.train:
        model = DecisionTreeClassifier(
            criterion = model_args.criterion,
            splitter = model_args.splitter,
            max_depth = model_args.max_depth,
            max_features = model_args.max_features,
            random_state = model_args.random_state,
            min_impurity_decrease = model_args.min_impurity_decrease,
            ccp_alpha = model_args.ccp_alpha,
        )

        # Fit the model
        model = model.fit(X_train_scaled, y_train)
    else:
        file = open(model_args.model_out_file, "rb")
        model = pickle.load(file)
        file.close
        
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Evaluation metrics
    if print_metrics:
        print(f"Test Accuracy: {test_acc}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}\n")

    # if model_args.train and len(model_args.model_out_file) > 0:
        # file = open(model_args.model_out_file, "wb+")
        # pickle.dump(model, file)
        # file.close

    return (model, train_acc, test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument('-m', '--model_out_file', default = "weights/dec_tree.csv", required = False) 
    parser.add_argument('-c', '--criterion', choices = ["gini", "entropy", "log_loss"], default = "gini", required = False)
    parser.add_argument('-s', '--splitter', choices = ["best", "random"], default = "best", required = False)
    parser.add_argument('-md', '--max_depth', type=int, default = None, required = False)
    parser.add_argument('-f', '--max_features', type=int, default = None, required = False)
    parser.add_argument('-r', '--random_state', type=int, default = None, required = False)
    parser.add_argument('-a', '--ccp_alpha', type=float, default = 0.0, required = False)
    parser.add_argument('-ts', '--test_size', type=float, default = 0.2, required = False)
    parser.add_argument('-i', '--min_impurity_decrease', type=float, default = 0.0, required = False)

    parser.add_argument('-t', '--train', action='store_true', required = False)
    parser.add_argument('--search', action='store_true', required = False)
    parser.add_argument('-d', '--debug', action='store_true', required = False)

    args = parser.parse_args()

    if (args.search):
        search(args.model_out_file)
    elif (args.debug):
        file = open(args.model_out_file, "rb")
        model = pickle.load(file)
        file.close

        print(f"Criterion: {model.criterion}")
        print(f"Splitter: {model.splitter}")
        print(f"Max Depth: {model.max_depth}")
        print(f"Max Features: {model.max_features}")
        print(f"CCP Alpha: {model.ccp_alpha}")
        print(f"Min Impurity Decrease: {model.min_impurity_decrease}")
        print(f"Random State: {model.random_state}")

        # print(export_text(model, feature_names=X.columns.values, class_names=["Non-Fraud", "Fraud"]))
    else:
        main(args, True)


