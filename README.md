# Winstars-AI-DS-internship

## Task 1 — MNIST Image Classification (OOP)

In this task, three machine learning models are implemented to classify handwritten digits from the MNIST dataset.

### Implemented models:
Random Forest
Feed-Forward Neural Network
Convolutional Neural Network

All models implement a common interface:

MnistClassifierInterface

Each model contains two required methods:

train() – trains the model
predict() – performs predictions

The models are wrapped in a unified class:

MnistClassifier

This class accepts the algorithm type as input:

cnn
rf
nn

This design ensures that the input/output format remains the same regardless of the chosen model.



## Task 2 — NER + Animal Image Classification Pipeline

### This task implements a machine learning pipeline combining NLP and Computer Vision.

The pipeline determines whether a text description of an animal matches the animal shown in an image.

Example:

Input:

Text: "There is a cow in the picture."
Image: cow.jpg

Output:

True
Pipeline Architecture

The solution consists of two models:

1. Named Entity Recognition (NER)

A Transformer-based model (DistilBERT) is trained to extract animal names from text.

Example:

"There is a dog in the picture"
→ detected entity: dog
2. Image Classification

A Convolutional Neural Network (CNN) is trained to classify animal images from a dataset containing 10 animal classes.
