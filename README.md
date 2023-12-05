# Number-Classifier
This repository contains a simple neural network implemented using TensorFlow, designed to recognize and predict numerical values. The model is trained on a dataset containing numbers from 1 to 10, and the architecture includes multiple dense layers for effective learning.

Key Features:

Training: The neural network is trained on a dataset with corresponding target values, teaching it to associate input numbers with their expected outputs.

Validation: During training, the model's performance is evaluated on a separate validation set to ensure it generalizes well to unseen data.

Testing: After training, the model undergoes a testing phase, where its accuracy is assessed on a distinct test set that was not used during training or validation.

Regression Task: As this is a regression problem, the model predicts continuous values rather than discrete classes. This makes it suitable for scenarios where predicting a numerical outcome is desired.

Usage:
The model can be easily trained and evaluated using the provided Python script. Feel free to adapt the architecture, hyperparameters, and data according to your specific requirements.

Dependencies:
TensorFlow: The machine learning library used for building and training the neural network.
Used Mnist dataset to train model, model inputs 28x28 pixels only, any  larger and it won't operate correctly.
Used Gradio version 3.50.2 and worked in venv environment.