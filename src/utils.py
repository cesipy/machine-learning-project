import random
import numpy as np

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


def count_frauds(data): 
    frauds = 0
    for row in data:
        if row[-1] == 1:
            frauds += 1
    return frauds