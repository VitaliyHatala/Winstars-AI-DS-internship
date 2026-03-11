import tensorflow as tf

from main_class import MnistClassifierInterface


class NeuralNetworkMnistClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = tf.keras.Sequential([

            tf.keras.layers.Flatten(input_shape=(28, 28)),

            tf.keras.layers.Dense(128, activation="relu"),

            tf.keras.layers.Dense(64, activation="relu"),

            tf.keras.layers.Dense(10, activation="softmax")

        ])

        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)

    def predict(self, X):
        predictions = self.model.predict(X)

        return predictions.argmax(axis=1)