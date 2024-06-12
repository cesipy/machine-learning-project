from neuronal_network_pure_numpy import NeuronalNet
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import count_frauds,  augment_train_data

    


def main():
    model_path = "weights/transactions-pure-numpy.pickle"
    df = pd.read_csv("data/transactions.csv")

    numpy_data = df.to_numpy()

    #labels = numpy_data[:, -1].astype(int)  # ensure labels are ints
    #train_data = numpy_data[:, :-1] / 255.0 # norm
    #print(labels)

    print("-------")


    #X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.3, random_state=10)
    X_train, X_test = train_test_split(numpy_data, test_size=0.3, random_state=10)

    X_train = augment_train_data(X_train)
    frauds_train = count_frauds(X_train)
    frauds_test  = count_frauds(X_test)
    print(f"frauds in train: {frauds_train}\nfrauds in test: {frauds_test}")

    print(f"train: {len(X_train)},\ntest {len(X_test)}")

    y_train = X_train[:, -1].astype(int)
    X_train = X_train[:, :-1] / 255.0   # norm
    y_test = X_test[:, -1].astype(int)
    X_test = X_test[:, :-1] / 255.0     # norm



    nn = NeuronalNet(learning_step_size=0.01, n_hidden_layers=10, iterations=500)
    nn.train(X_train.T, y_train)
    nn.save_model(model_path)
    
    nn2 = NeuronalNet()
    nn2.load_model(model_path)
    preds = nn2.predict(X_test.T)

    accuracy = nn2.get_accuracy(preds, y_test)
    print(f"Final Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()
