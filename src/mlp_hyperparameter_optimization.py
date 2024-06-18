import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
from utils import count_frauds, augment_train_data
from logger import Logger

logger = Logger()

BATCH_SIZE = 32

class TransactionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransactionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def init_model(input_dim, hidden_dim):
    model = TransactionModel(input_dim, hidden_dim)
    return model

def objective(trial):
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

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    
    # hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 10, 50)
    
    model = init_model(input_dim, hidden_dim)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        logger.log(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    logger.log(f"Test Accuracy: {test_acc:.4f}")

    return test_acc

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    logger.log("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    logger.log("  Value: {}".format(trial.value))

    print("  Params: ")
    logger.log("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        logger.log("    {}: {}".format(key, value))
