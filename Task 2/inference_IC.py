import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model_path = r"D:\STUDY\PyCharm\PyCharm 2025.3.1.1\Projects\PythonProject3\animal_classifier_cnn.h5"
model = load_model(model_path)


data_dir = r"C:\Users\user\Desktop\Datasets\Dataset_Animals\raw-img"
class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
class_names.sort()


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Test

test_image_path = r"C:\Users\user\Desktop\Datasets\Dataset_Animals\raw-img\butterfly\e83db70e2cf0073ed1584d05fb1d4e9fe777ead218ac104497f5c97faeebb5bb_640.png"
img = preprocess_image(test_image_path)
pred = model.predict(img)
pred_class = np.argmax(pred, axis=1)[0]
print(f"Predicted class for {os.path.basename(test_image_path)}: {class_names[pred_class]}")
