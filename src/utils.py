import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def augment_train_data(X_train, factor=100):
    X_train = X_train.tolist()  # numpy vs python list - append not working for np
    frauds = []
    for row in X_train:
        if row[-1] == 1:
            frauds.append(row)
    initial_frauds = len(frauds)
    current_frauds = initial_frauds

    while current_frauds < (initial_frauds * factor):
        random_index = random.randint(0, len(frauds) - 1)
        X_train.append(frauds[random_index])
        current_frauds += 1
    
    random.shuffle(X_train)
    X_train = np.array(X_train)  # back to np
    new_frauds_count = count_frauds(X_train)
    print(f"new amount of frauds {new_frauds_count}")
    return X_train

def preprocess_data(dataset: np.ndarray, factor=10, test_size=0.3, random_state=10):
    # standardize the data as normal distribution
    scaler = StandardScaler()
    dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])

    X_train, X_test = train_test_split(dataset, test_size=test_size, random_state=random_state)

    # perform data augmentation
    X_train = augment_train_data(X_train, factor=factor)
    frauds_train = count_frauds(X_train)
    frauds_test = count_frauds(X_test)
    print(f"Frauds in train: {frauds_train}\nFrauds in test: {frauds_test}")

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Separate features and labels
    y_train = X_train[:, -1].astype(int)
    X_train = X_train[:, :-1]
    y_test = X_test[:, -1].astype(int)
    X_test = X_test[:, :-1]

    return X_train, X_test, y_train, y_test


def count_frauds(data): 
    frauds = 0
    for row in data:
        if row[-1] == 1:
            frauds += 1
    return frauds