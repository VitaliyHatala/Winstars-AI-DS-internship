import tensorflow as tf
from main_class import MnistClassifierInterface

class CNNMnistClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, 28, 28, 1)
        self.model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32
        )

    def predict(self, X):
        X = X.reshape(-1, 28, 28, 1)
        predictions = self.model.predict(X)
        return predictions.argmax(axis=1)
