import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import confusion_matrix

data_dir = "C:\\Users\\user\\Desktop\\Datasets\\Dataset_Animals\\raw-img"
images = []
labels = []
class_names = []

for category in os.listdir(data_dir):
    class_names.append(category)

for n, category in enumerate(class_names):
    for file in os.listdir(os.path.join(data_dir, category)):
        img = cv2.imread(os.path.join(data_dir, category, file))
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_color, (128, 128))
        images.append(img_resized)
        labels.append(n)

images = np.array(images) / 255.0
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

train_labels = to_categorical(train_labels, num_classes=len(class_names))
test_labels = to_categorical(test_labels, num_classes=len(class_names))

print("Model is studying...")

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1)

predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)
test_labels_argmax = np.argmax(test_labels, axis=1)

model.save("animal_classifier_cnn.h5")

print(confusion_matrix(test_labels_argmax, predictions))



