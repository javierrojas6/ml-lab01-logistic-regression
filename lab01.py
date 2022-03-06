# %% Librerias
from black import out
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

XFlattened = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
labels = sorted(set(Y))

print("X.shape", XFlattened.shape)
print("Labels", labels)

# %% draw number
## draw number
def drawNumber(numberArray, label):
    plt.title("Number " + str(label))
    plt.imshow(np.asarray(numberArray), cmap="gray", vmin=0, vmax=np.max(numberArray))
    plt.show


# %% initialize all models
# Initialize models
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
# Saves the model on a file
def saveModel(models, modelFileName):
    buffer = str(len(labels))
    for model in models:
        buffer += "\n" + model.label + " " + str(model)
    out = open(modelFileName, "w")
    out.write(buffer)
    out.close()


# %% train the model
# train the model
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

# %% Load models using a file with its weights
# Load models using a file with its weights
def loadModels(filename):
    models_file = open(filename, "r")
    models_lines = models_file.readlines()
    models_file.close()

    labels = []
    models = []
    for l in models_lines[1:]:
        d = l.split()
        labels += [d[0]]
        model = PUJ.Model.Logistic()
        model.label = str(d[0])
        model.setParameters([float(v) for v in d[2:]])
        models += [model]

    return models

# %% evaluate all trained models
# evaluate all trained models
def evaluateAll(models, X):
    estimatedLabels = []
    for row in X:
        estimatedLabels += [evaluate(models, row)]
    return np.array(estimatedLabels)


# %% evaluate one trained model
# evaluate one trained model
def evaluate(models, image):
    flatImage = image.reshape((image.shape[0] * image.shape[1]))
    results = []
    for model in models:
        results += [model.evaluate(flatImage)[0, 0]]

    return results.index(max(results))

# %% generate confusion matrix
# generate confusion matrix
def generateConfusionMatrix(realY, estimatedY, size=2):
    """
    Generate a confusion matrix

    :param realY: The actual labels of the data
    :param estimatedY: the estimated labels of the data points
    :param size: the number of classes, defaults to 2 (optional)
    """
    matrix = np.zeros((size, size))
    for i in range(realY.shape[0]):
        matrix[int(realY[i]), int(estimatedY[i])] += 1

    return matrix

# %% show all metric in all evaluation
# show all metric in all evaluation
def metrics(y_real, y_estimated, labels):
    cm = confusion_matrix(y_real, y_estimated, labels=labels)
    cr = classification_report(y_real, y_estimated, labels=labels)
    return cm, cr


# %% Main flow

# # load random weight in a new model
# models = initializeModel(labels, XFlattened.shape[1], seed=12)

# # save the initial weight in a file
# saveModel(models, MODEL_WEIGHTS)

# # split randonly the dataset into train and test dataset
# X_train, X_test, y_train, y_test = train_test_split(XFlattened, Y, train_size=0.7, shuffle=True)

# # train all models
# trainedModels = trainModels(models, X_train, y_train)

# # save the new weight into a file
# saveModel(trainedModels, "mnists_weights_trained.txt")

# load the weights from the file
models = loadModels("./trained_weights/mnists_weights_trained_10000.txt")

# evaluate the model usin the test dataset
estimatedY = evaluateAll(models, x_test)

# generate the metric using the results from test dataset
cm, cr = metrics(y_test, estimatedY, labels)

# print the metrics
print(cm)
print(cr)
