import numpy as np
import matplotlib.pyplot as plt
import os
import feature_extractor as im_f
import inspect_data as insp


dataset = insp.make_dataset('road-signs')

train_imgs, train_labels, test_imgs, test_labels, classes = dataset
#print(classes)

def features_function(file):
    """return all functions from file that are not private or from other libraries

    Parameters
    ----------
        file : file
            file with functions

    Returns
    ------- 
        function_names : list
            list of functions
    """
    function_names = [getattr(file, func) for func in dir(file) if not func.startswith('__') and not func.startswith('_') and not func.startswith('np') and not func.startswith('insp') and not func.startswith('plt') and not func.startswith('os') and not func.startswith('torch') and not func.startswith('device') and not func.startswith('To') and not func.startswith('read') and not func.startswith('make')]
    return function_names

def process(data, labels, function):
    """process data with function

    Parameters
    ----------
        data : numpy.ndarray
            data to be processed
        labels : numpy.ndarray
            labels of data
        function : function
            function to be applied to data

    Returns
    -------
        X : numpy.ndarray
            processed data
        Y : numpy.ndarray
            labels of data
    """
    all_features = []
    all_labels = labels

    for imgs in data:
        features = function(imgs)
        features = features.reshape(-1)
        all_features.append(features)
    
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


if __name__ == "__main__":


    for function in features_function(im_f):
        print(function.__name__)
        X, Y = process(train_imgs, train_labels, function)
        print(X.shape)
        print(Y.shape)
        data = np.concatenate([X, Y[:, None]], 1)

        np.savetxt('low_features/'+str(function.__name__)+'_train.txt.gz', data)

        X, Y = process(test_imgs, test_labels, function)
        print(X.shape)
        print(Y.shape)
        data = np.concatenate([X, Y[:, None]], 1)

        np.savetxt('low_features/'+str(function.__name__)+'_test.txt.gz', data)

    


