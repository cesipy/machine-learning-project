import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import utils
import matplotlib.pyplot as plt
import pickle
import argparse

# hyperparameter learned using optuna package
# tuned params: hidden_dim, learning_rate and epochs
# Test Accuracy: 0.9989
# Best trial:
#   Value: 0.9994148111302923
#   Params: 
#     hidden_dim: 94
#     lr: 0.0004301150216793739
#     epochs: 45

EPOCHS        = 45
BATCH_SIZE    = 32
HIDDEN_DIM    = 94
LEARNING_RATE = 0.0004301150216793739
FACTOR        = 100
WEIGHTS_PATH  = "weights/mlp.pkl"


def plot_metrics(losses, accuracies):
    epochs = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))

    #loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    #acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color='r')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

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

    @staticmethod
    def initialize_datasets(X_train, y_train, X_test, y_test):
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, test_loader
    
    def train_model(self, train_loader, optimizer, criterion):
        losses = []
        accuracies = []
        for epoch in range(EPOCHS):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.7f}, Accuracy: {epoch_acc:.7f}")

        return losses, accuracies
    
    def save_model_pkl(self, path:str):
        with open(path, "wb") as file:
            pickle.dump(self.state_dict(), file)
        
    def load_model_pkl(self, path: str):
        with open(path, "rb") as file:
            state_dict = pickle.load(file)
            self.load_state_dict(state_dict)

def init_model(input_dim, hidden_dim):
    model = TransactionModel(input_dim, hidden_dim)
    return model

def main(args):
    df = pd.read_csv("data/transactions.csv")
    numpy_data = df.to_numpy()

    X_train, X_test, y_train, y_test = utils.preprocess_data(numpy_data, factor=FACTOR)

    input_dim = X_train.shape[1]
    model = init_model(input_dim, HIDDEN_DIM)

    #init dataset loaders
    train_loader, test_loader = TransactionModel.initialize_datasets(X_train, y_train, X_test, y_test)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
     
    if args.train:
        losses, accuracies = model.train_model(train_loader, optimizer, criterion)

        model.save_model_pkl(WEIGHTS_PATH)
        
        
    model.load_model_pkl(WEIGHTS_PATH)
    


    # evaluation
    model.eval()
    correct = 0
    total = 0
    test_outputs = []
    test_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            test_outputs.extend(outputs[:, 1].cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.7f}")

    test_outputs = np.array(test_outputs)
    test_labels = np.array(test_labels)
    test_roc_auc = roc_auc_score(test_labels, test_outputs)
    print(f"Test ROC AUC Score: {test_roc_auc:.7f}")

    if args.train:
        plot_metrics(losses=losses, accuracies=accuracies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()
    main(args)