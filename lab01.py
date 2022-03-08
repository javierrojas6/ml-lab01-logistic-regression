# %% Librerias
from black import out
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

# libreria local
import PUJ.Model.Logistic

# %% constants
# constants

MODEL_WEIGHTS = "mnists_weights.txt"

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
        opt.setNumberOfIterations(5000)
        opt.setNumberOfDebugIterations(100)
        opt.Fit()

        models[i] = model

    return models


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


def evaluateAll(models, X):
    estimatedLabels = []
    for row in X:
        estimatedLabels += [evaluate(models, row)]
    return np.array(estimatedLabels)


def evaluate(models, image):
    results = []
    for model in models:
        results += [model.evaluate(image)[0, 0]]

    return results.index(max(results))


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


def metrics(y_real, y_estimated, labels):
    cm = confusion_matrix(y_real, y_estimated, labels=labels)
    cr = classification_report(y_real, y_estimated, labels=labels)
    return cm, cr

# %% download MINST dataset
# download MINST dataset
tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)
labels = sorted(set(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)

#aplanar y estandarizar entrenamiento
XFlattened_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
XFlattened_train = preprocessing.scale(XFlattened_train)

#aplanar y estandarizar test
XFlattened_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
XFlattened_test = preprocessing.scale(XFlattened_test)

models = initializeModel(labels, XFlattened_train.shape[1], seed=12)
saveModel(models, MODEL_WEIGHTS)

trainedModels = trainModels(models, XFlattened_train, y_train)
saveModel(trainedModels, "./trained_weights/mnists_weights.txt")

models = loadModels("./trained_weights/mnists_weights.txt")

print("METRICAS EN ENTRENAMIENTO")
estimatedY = evaluateAll(models, XFlattened_train)
cm, cr = metrics(y_train, estimatedY, labels)

print(cm)
print(cr)

print("METRICAS EN TEST")
estimatedY = evaluateAll(models, XFlattened_test)
cm, cr = metrics(y_test, estimatedY, labels)

print(cm)
print(cr)