
from utils import count_frauds, augment_train_data

from dataclasses import dataclass
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
    train_data_file : str
    model_out_file : str
    criterion : str = "gini"
    splitter : str = "best"
    max_depth : int = None
    max_features : int = None
    random_state : int = None
    ccp_alpha : float = 0.0
    min_impurity_decrease : float = 0.0
    train : bool = False

def main(model_args: ModelArgs):
    train_data = pd.read_csv(model_args.train_data_file)

    X = train_data.drop(columns = ["Class", "Time"])
    Y = train_data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        model.fit(X_train_scaled, y_train)

        # fig = pyplot.figure(figsize = (25, 20))
        # _ = plot_tree(model) feature_names = X.columns.values)
    else:
        file = open(model_args.model_out_file, "rb")
        model = pickle.load(file)
        file.close
        
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    if model_args.train:
        file = open(model_args.model_out_file, "wb+")
        pickle.dump(model, file)
        file.close

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument('-d', '--train_data_file', default = "data/transactions.csv", required = False) 
    parser.add_argument('-m', '--model_out_file', default = "weights/dec_tree.csv", required = False) 
    parser.add_argument('-c', '--criterion', choices = ["gini", "entropy", "log_loss"], default = "gini", required = False)
    parser.add_argument('-s', '--splitter', choices = ["best", "random"], default = "best", required = False)
    parser.add_argument('-md', '--max_depth', type=int, default = None, required = False)
    parser.add_argument('-f', '--max_features', type=int, default = None, required = False)
    parser.add_argument('-r', '--random_state', type=int, default = None, required = False)
    parser.add_argument('-a', '--ccp_alpha', type=float, default = 0.0, required = False)
    parser.add_argument('-i', '--min_impurity_decrease', type=float, default = 0.0, required = False)
    parser.add_argument('-t', '--train', action='store_true', required = False)

    args = parser.parse_args()

    main(args)

