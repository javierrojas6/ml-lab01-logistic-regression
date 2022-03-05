# %% Librerias
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# libreria local
import PUJ.Model.Logistic

# %% constants
# constants

MODEL_WEIGHTS = "mnists_weights.txt"

# %% download MINST dataset
# download MINST dataset
tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)
X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
labels = sorted(set(Y))

print("X.shape", X.shape)
print("Labels", labels)

# %% draw number
## draw number
def drawNumber(numberArray, label):
    plt.title("Number " + str(label))
    plt.imshow(np.asarray(numberArray), cmap="gray", vmin=0, vmax=np.max(numberArray))
    plt.show


# %% initialize all models
def initializeModel(labels, paramsSize, seed=0):
    if seed > 0:
        random.seed(seed)

    # It creates a list of models, one for each label.
    models = []
    for labelItem in labels:
        model = PUJ.Model.Logistic()
        model.label = str(labelItem)
        # Initializing the parameters of the model with random values.
        model.setParameters([random.uniform(-1, 1) for n in range(paramsSize + 1)])
        models += [model]

    return models


# %% Saves the model
def saveModel(models, modelFileName):
    buffer = str(len(labels))
    for model in models:
        buffer += "\n" + model.label + " " + str(model)
    out = open(modelFileName, "w")
    out.write(buffer)
    out.close()


# %% train the model
def trainModels(models, x, y, learningRate=1e-3):
    for i, model in enumerate(models):

        print("\ntraining model: ", model.label)

        yTemp = np.matrix(np.array([1 if item == int(model.label) else 0 for item in y]))
        modelCost = PUJ.Model.Logistic.Cost(model, x, yTemp.T)
        # # Debugger
        debugger = PUJ.Optimizer.Debug.Simple
        # debugger = PUJ.Optimizer.Debug.PlotPolynomialCost(x, yTemp.T)

        # Fit using an optimization algorithm
        opt = PUJ.Optimizer.GradientDescent(modelCost)
        opt.setDebugFunction(debugger)
        opt.setLearningRate(learningRate)
        opt.setNumberOfIterations(200)
        opt.setNumberOfDebugIterations(10)
        opt.Fit()

        models[i] = model

    return models


# %% Main
models = initializeModel(labels, X.shape[1], seed=12)
saveModel(models, MODEL_WEIGHTS)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)

print("X_train: ", X_train.shape)
trainedModels = trainModels(models, X_train, y_train)
saveModel(trainedModels, "mnists_weights2.txt")


# %% cost function
# cost = PUJ.Model.Logistic.Cost(model, x_train, y_train)
