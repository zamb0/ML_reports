import numpy as np
import pvml
import matplotlib.pyplot as plt
import os

def nothing(X, dumb = None):
    return X, None

def meanvar(X, mu=None, std=None):
    if mu is None and std is None:
        mu = X.mean(0)
        std = X.std(0)

    X = (X - mu) / std
    return X, mu, std

def minmax(X, xmin=None, xmax=None):
    if xmin is None and xmax is None:
        xmin = X.min(0)
        xmax = X.max(0)
    X = (X - xmin) / (xmax - xmin) 

    return X, xmin, xmax

def maxabs(X, amax=None):
    if amax is None:
        amax = np.abs(X).max(0)
    X = X / amax

    return X, amax

def whitening(X, mu=None, w=None):
    if mu is None and w is None:
        mu = X.mean(0)
        sigma = np.cov(X.T)
        evals , evecs = np.linalg.eigh(sigma) 
        w = evecs / np.sqrt(evals)

    X = (X - mu) @ w 

    return X, mu, w

def l2(X, dumb = None):
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    q = np.maximum(q, 1e-15)
    return X, None

def l1(X, dumb = None):
    q = np.abs(X).sum(1, keepdims=True)
    q = np.maximum(q, 1e-15)
    return X, None

def is_vector(variable):
    return isinstance(variable, np.ndarray) and variable.ndim == 1

def is_list(variable):
    return isinstance(variable, list)


def try_all_normalization(Xtrain, Ytrain, Xtest, Ytest):
    norm_func = [(nothing, []),
             (meanvar, [None, None]),
             (minmax, [None, None]),
             (maxabs, [None]),
             (whitening, [None, None]),
             (l2, []),
             (l1, [])]

    for func, args in norm_func:
        Xtrain, *args = func(Xtrain, *args)
        Xtest, *_ = func(Xtest, *args)

        m = Xtrain.shape[0]
        network = pvml.MLP([1600, 35])
        results = []

        for epoch in range(30):
            network.train(Xtrain, Ytrain, lr=1e-4, steps=m//40, batch=40)

            predictions, logits = network.inference(Xtrain)
            train_acc = (predictions == Ytrain).mean()

            predictions, logits = network.inference(Xtest)
            test_acc = (predictions == Ytest).mean()

            print("Epoch %d: train=%.3f validation=%.3f" % (epoch + 1, train_acc, test_acc))
            results.append([train_acc, test_acc])

        network.save("norm_mdl/"+str(func.__name__)+"_mlp.npz")
        np.savetxt("norm_mdl/"+str(func.__name__)+"_res.txt", results, fmt="%.6f")
        
        #print([arg for arg in args])
        if len(args) > 0 and args is not None:
            np.savetxt("norm_mdl/"+str(func.__name__)+"_param.txt", np.vstack([arg for arg in args]), fmt="%s")
        else:
            np.savetxt("norm_mdl/"+str(func.__name__)+"_param.txt", ["None"], fmt="%s")


def try_batch(Xtrain, Ytrain, Xtest, Ytest):
    M = Xtrain.shape[0]

    batch_size = [M, 1, 10, 20, 40, 80, 160, 320, 640]

    for m in batch_size:
        network = pvml.MLP([1600, 35])
        result = []
        for epoch in range(30):
            network.train(Xtrain, Ytrain, lr=1e-4, steps=M//m, batch=m)

            predictions, logits = network.inference(Xtrain)
            train_acc = (predictions == Ytrain).mean()

            predictions, logits = network.inference(Xtest)
            test_acc = (predictions == Ytest).mean()

            print("Epoch %d: train=%.3f validation=%.3f" % (epoch + 1, train_acc, test_acc))
            result.append([train_acc, test_acc])

        network.save("epoch_mdl/"+str(m)+"_mlp.npz")
        np.savetxt("epoch_mdl/"+str(m)+"_res.txt", result, fmt="%.6f")

    np.savetxt("epoch_mdl/param.txt", [mu, std], fmt="%.6f")

def try_layer(Xtrain, Ytrain, Xtest, Ytest):
    M = Xtrain.shape[0]
    i=0
    layers = [[1600, 35], [1600, 1000, 35], [1600, 1000, 500, 35], [1600, 1000, 500, 250, 35], [1600, 1000, 500, 250, 100, 35]]
    for layer in layers:
        network = pvml.MLP(layer)
        result = []
        print(layer)
        for epoch in range(30):
            network.train(Xtrain, Ytrain, lr=1e-4, steps=M//40, batch=40)

            predictions, logits = network.inference(Xtrain)
            train_acc = (predictions == Ytrain).mean()

            predictions, logits = network.inference(Xtest)
            test_acc = (predictions == Ytest).mean()
            
            print("Epoch %d: train=%.3f validation=%.3f" % (epoch + 1, train_acc, test_acc))
            result.append([train_acc, test_acc])

        network.save("layer_mdl/"+str(i)+"_mlp.npz")
        np.savetxt("layer_mdl/"+str(i)+"_res.txt", result, fmt="%.6f")
        i+=1

    np.savetxt("layer_mdl/param.txt", [mu, std], fmt="%.6f")


mu, std = None, None
xmin, xmax = None, None
amax = None
mu, w = None, None

words = open("speech_comands/classes.txt").read().split()
#print(words)

data = np.load("speech_comands/train.npz")
Xtrain = data["arr_0"]
Ytrain = data["arr_1"]
#print(Xtrain.shape, Ytrain.shape)

data = np.load("speech_comands/validation.npz")
Xtest = data["arr_0"]
Ytest = data["arr_1"]
#print(Xtest.shape, Ytest.shape)

#try_all_normalization(Xtrain, Ytrain, Xtest, Ytest)

Xtrain, mu, std = meanvar(Xtrain)
Xtest, *_ = meanvar(Xtest, mu, std)

#try_batch(Xtrain, Ytrain, Xtest, Ytest)
#try_layer(Xtrain, Ytrain, Xtest, Ytest)





    
    



