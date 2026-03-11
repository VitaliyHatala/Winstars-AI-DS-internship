# Task 2 — Animal Verification Pipeline (NER + Image Classification)
## Overview

This task implements a machine learning pipeline that combines Natural Language Processing (NLP) and Computer Vision (CV).

The goal of the system is to determine whether a text description of an animal matches the animal shown in an image.

Example:

Input

Text: "There is a cow in the picture."
Image: cow.jpg

Output

True

The pipeline extracts the animal name from the text using Named Entity Recognition (NER) and compares it with the predicted animal from the image classification model.

Pipeline Architecture

The solution consists of two independent models combined into a single pipeline.

### 1. Named Entity Recognition (NER)

A transformer-based model (DistilBERT) is trained to extract animal names from text.

Example:

Input text:
"There is a dog in the image"

NER output:
dog

The model is trained using the Hugging Face Transformers library.

Main scripts:

train_NER.py
inference_NER.py

### 2. Animal Image Classification

A Convolutional Neural Network (CNN) is trained to classify animal images.

Dataset requirements:

at least 10 animal classes

images are resized and normalized before training

Main scripts:

train_image_classifier.py
inference_image_classifier.py

The model predicts the animal category present in the input image.
