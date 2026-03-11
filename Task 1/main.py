from CNN import CNNMnistClassifier
from Neural_Network import NeuralNetworkMnistClassifier
from Random_Forest import RandomForestMnistClassifier


class MnistClassifier:

    def __init__(self, algorithm):

        if algorithm == "cnn":
            self.model = CNNMnistClassifier()

        elif algorithm == "nn":
            self.model = NeuralNetworkMnistClassifier()

        elif algorithm == "rf":
            self.model = RandomForestMnistClassifier()

        else:
            raise ValueError("Algorithm must be 'cnn', 'nn', or 'rf'")

    def train(self, X_train, y_train):

        self.model.train(X_train, y_train)

    def predict(self, X):

        return self.model.predict(X)