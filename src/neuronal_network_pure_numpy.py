import numpy as np
#import mlx.core as mx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

ITERATIONS = 1500

class NeuronalNet:

    def __init__(self, n_hidden_layers= 100, learning_step_size=0.15, iterations=1000): 
        self.n_hidden_layers    = n_hidden_layers
        self.learning_step_size = learning_step_size
        self.iterations         = iterations
        self.parameters         = {}
        self.losses             = []

    def leaky_ReLU(self, x, alpha=0.01): 
        return np.maximum(alpha*x, x)

    def dleaky_ReLU(self, x, alpha=0.01):
        """
        derivative of leaky ReLU
        """ 
        return np.where(x>0, 1, alpha) 

    def ReLU(self, x):
        return np.maximum(0, x)

    def dReLU(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max value for numerical stability
        return expZ / np.sum(expZ, axis=0, keepdims=True)


    def one_hot(self, y, num_classes):
        one_hot_y = np.zeros((num_classes, y.size))
        one_hot_y[y, np.arange(y.size)] = 1
        return one_hot_y


    def forward(self, W_1, W_2, b_1, b_2, input_data):
        Z_1 = np.dot(W_1, input_data) + b_1
        A_1 = self.leaky_ReLU(Z_1)
        Z_2 = np.dot(W_2, A_1) + b_2
        A_2 =  self.softmax(Z_2)

        return Z_1, A_1, Z_2, A_2

    def backward(self, Z_1, A_1, Z_2, A_2, W_1, W_2, b_1, b_2, input_data, labels):
        x_shape, m = input_data.shape 
        y = self.one_hot(labels, A_2.shape[0])
        dZ_2 = A_2 - y
        dW_2 = 1/m * np.dot(dZ_2, A_1.T)
        db_2 = 1/m * np.sum(dZ_2, axis=1, keepdims=True)
        dA_1 = np.dot(W_2.T, dZ_2)
        dZ_1 = dA_1 * self.dleaky_ReLU(Z_1)
        dW_1 = 1/m * np.dot(dZ_1, input_data.T)
        db_1 = 1/m * np.sum(dZ_1)

        return dW_1, db_1, dW_2, db_2

    def init_parameter(self, input_shape, hidden_shape, output_shape):
        W1 = np.random.randn(hidden_shape, input_shape) * np.sqrt(2. / input_shape)
        b1 = np.zeros((hidden_shape, 1))
        W2 = np.random.randn(output_shape, hidden_shape) * np.sqrt(2. / hidden_shape)
        b2 = np.zeros((output_shape, 1))
        return W1, W2, b1, b2

    def update_parameter(self, W_1, W_2, b_1, b_2, dW_1, db_1, dW_2, db_2, step_size):
        W_1 -= dW_1 * step_size
        b_1 -= db_1 * step_size
        W_2 -= dW_2 * step_size
        b_2 -= db_2 * step_size

        self.parameters = {
            "w1": W_1,
            "w2": W_2, 
            "b1": b_1, 
            "b2": b_2
            }

        return W_1, W_2, b_1, b_2

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)


    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def compute_loss(self, A2, y):
        epsilon = 1e-10  # small value to prevent log(0)
        size = y.size
        log_probs = -np.log(A2[y, np.arange(size)] + epsilon)  # add epsilon to avoid log(0)
        loss = np.sum(log_probs) / size
        return loss

    def train(self, input_data, labels):
        n_input = input_data.shape[0]
        n_output = len(np.unique(labels))
        counter = 0
        W_1, W_2, b_1, b_2 = self.init_parameter(n_input, self.n_hidden_layers, n_output)

        try: 
            while counter < self.iterations:
                Z_1, A_1, Z_2, A_2 =  self.forward(W_1, W_2, b_1, b_2, input_data)
                dW_1, db_1, dW_2, db_2 = self.backward(Z_1, A_1, Z_2, A_2, W_1, W_2, b_1, b_2, input_data, labels)
                W_1, W_2, b_1, b_2 = self.update_parameter(W_1, W_2, b_1, b_2, dW_1, db_1, dW_2, db_2, self.learning_step_size)
                if counter % 10 == 0:
                    predictions = self.get_predictions(A_2)
                    accuracy    = self.get_accuracy(predictions, labels)
                    loss = self.compute_loss(A_2, labels)
                    self.losses.append(loss)
                    print(f"Iteration {counter:4}: Loss {loss:.3f}, Accuracy {accuracy:.3f}")
                    
                counter += 1

            return W_1, W_2, b_1, b_2
        except KeyboardInterrupt:
            print("Training interrupted")
            return W_1, W_2, b_1, b_2
    
    def predict(self, input_data): 
        if self.parameters == {}: 
            raise ValueError("Model not trained yet")
        # load params
        W_1 = self.parameters["w1"]
        W_2 = self.parameters["w2"]
        b_1 = self.parameters["b1"]
        b_2 = self.parameters["b2"]

        Z_1, A_1, Z_2, A_2 =  self.forward(W_1, W_2, b_1, b_2, input_data)  # only A2 is important
        preds = self.get_predictions(A_2)
        return preds

    def save_model(self, file_path):
        file = open(file_path, "wb")
        pickle.dump(self.parameters, file)
        file.close

    def load_model(self, file_path): 
        file = open(file_path, "rb")
        self.parameters = pickle.load(file)
        file.close

def plot_loss(losses): 
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over iterations")
    plt.show()


def main(): 
    model_path = "weights/weights-1layer-500.pickle"
    df = pd.read_csv("data/train.csv")
    train_data = df.to_numpy()
    
    labels = train_data[:,0]
    data   = train_data[:, 1:] / 255.0

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=10)

    nn = NeuronalNet(learning_step_size=0.15, n_hidden_layers=500, iterations=ITERATIONS)
    nn.train(X_train.T, y_train)
    plot_loss(nn.losses)
    nn.save_model(model_path)

    nn2 = NeuronalNet()
    nn2.load_model(model_path)
    preds = nn2.predict(X_test.T)

    accuracy = nn2.get_accuracy(preds, y_test)
    print(f"Final Accuracy: {accuracy:.2f}")





if __name__== '__main__': 
    main()
