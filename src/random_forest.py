from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from utils import augment_train_data, count_frauds
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(): 
    df = pd.read_csv("data/transactions.csv")
    numpy_data = df.to_numpy()

    X_train, X_test = train_test_split(numpy_data, test_size=0.3, random_state=10)

    X_train = augment_train_data(X_train, factor=50)
    frauds_train = count_frauds(X_train)
    frauds_test  = count_frauds(X_test)
    print(f"frauds in train: {frauds_train}\nfrauds in test: {frauds_test}")

    print(f"train: {len(X_train)},\ntest {len(X_test)}")

    y_train = X_train[:, -1].astype(int)
    X_train = X_train[:, :-1] 
    y_test = X_test[:, -1].astype(int)
    X_test = X_test[:, :-1] 

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)


    y_pred = random_forest.predict(X_test)

    # validation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")



if __name__ == '__main__':
    main()
