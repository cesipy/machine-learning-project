from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from utils import augment_train_data, count_frauds
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# hyperparameter learned using optuna package
#finetuned params: factor (for data augmentation), n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf
#Best trial:
#  Value: 0.9995903677912046
#  Params:
#    factor: 32
#    n_estimators: 52
#    max_depth: 17
#    max_features: sqrt
#    min_samples_split: 3
#    min_samples_leaf: 1

def main(): 
    df = pd.read_csv("data/transactions.csv")
    numpy_data = df.to_numpy()

    X_train, X_test = train_test_split(numpy_data, test_size=0.3, random_state=10)

    X_train = augment_train_data(X_train, factor=32)
    frauds_train = count_frauds(X_train)
    frauds_test  = count_frauds(X_test)
    print(f"frauds in train: {frauds_train}\nfrauds in test: {frauds_test}")

    print(f"train: {len(X_train)},\ntest {len(X_test)}")

    y_train = X_train[:, -1].astype(int)
    X_train = X_train[:, :-1] 
    y_test = X_test[:, -1].astype(int)
    X_test = X_test[:, :-1] 

    random_forest = RandomForestClassifier(n_estimators=42, 
                                           max_depth=17, 
                                           max_features="sqrt",
                                           min_samples_split=3,
                                           min_samples_leaf=1,
                                           )
    random_forest.fit(X_train, y_train)


    y_pred = random_forest.predict(X_test)

    # validation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")



if __name__ == '__main__':
    main()
