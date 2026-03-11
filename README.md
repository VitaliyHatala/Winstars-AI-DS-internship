# Winstars-AI-DS-internship

Task 1 — MNIST Image Classification (OOP)

In this task, three machine learning models are implemented to classify handwritten digits from the MNIST dataset.

Implemented models:

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
