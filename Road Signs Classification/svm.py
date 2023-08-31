import numpy as np
import matplotlib.pyplot as plt
import inspect_data as insp
import os
import itertools as it
from sklearn.svm import SVC
from utility import conf_mat, display_conf_mat
from mlp import LowFeaturesTest, ToTensor
from torch.utils.data import DataLoader

def svm_train_sklearn(X, Y, c):
    """Train a SVM classifier using sklearn 

    Parameters
    ----------
        X : numpy array 
            Training data
        Y : numpy array
            Training labels
        c : float    
            Penalty parameter C of the error term.

    Returns
    -------
        clf : sklearn.svm.SVC
            Trained SVM classifier

    """
    clf = SVC(kernel='rbf', C=c, probability=True)
    clf.fit(X, Y)
    return clf

def svm_inference_sklearn(X, clf):
    """Inference using a SVM classifier using sklearn

    Parameters
    ----------
        X : numpy array
            Test data
        clf : sklearn.svm.SVC
            Trained SVM classifier

    Returns
    -------
        Y : numpy array
            Predicted labels
    
    """
    
    return clf.predict(X)
   

def train_svm(features, c):
    """Train a SVM classifier using sklearn

    Parameters
    ----------
        features : list
            List of numpy arrays containing the training features, test features, training labels and test labels
        c : float
            Penalty parameter C of the error term.

    Returns
    -------
        train_acc : float
            Training accuracy
        test_acc : float
            Test accuracy
        clf : sklearn.svm.SVC
            Trained SVM classifier

    """
    train_features, test_features, train_labels, test_labels = features

    #W, b = one_vs_rest_train(train_features, train_labels, lambda_, lr, steps)
    #clf = one_vs_rest_train_sklearn(train_features, train_labels)
    clf = svm_train_sklearn(train_features, train_labels, c)

    #train_pred = one_vs_rest_inference(train_features, W, b)
    train_pred = svm_inference_sklearn(train_features, clf)
    train_acc = (train_pred == train_labels).mean()
    print(f"train_acc = {train_acc*100:.5f}")

    #test_pred = one_vs_rest_inference(test_features, W, b)
    test_pred = svm_inference_sklearn(test_features, clf)
    test_acc = (test_pred == test_labels).mean()
    print(f"test_acc = {test_acc*100:.5f}")

    return np.array(train_acc), np.array(test_acc), clf


def feature_n_combination(n=2, features=['color_histogram', 'cooccurrence_matrix', 'edge_direction_histogram', 'rgb_cooccurrence_matrix'], c=1e-6):
    """Train a SVM classifier using sklearn

    Parameters
    ----------
        n : int
            Number of features to combine
        features : list
            List of features to combine
        c : float
            Penalty parameter C of the error term.

    Returns
    -------
        train_acc_list : numpy array
            List of training accuracies
        test_acc_list : numpy array
            List of test accuracies
        comb_list : numpy array
            List of feature combinations

    """
    iters = it.combinations(features, n)

    train_acc_list = np.array([]) 
    test_acc_list = np.array([])
    comb_list = np.array([])
    
    for iter in iters:
        X = np.array([])
        X_test = np.array([])
        Y = np.array([])
        Y_test = np.array([])
        for its in iter:
            X_temp, X_test_temp, Y_temp, Y_test_temp, _ = insp.import_features(its)
            if X.size != 0:
                X = np.concatenate((X, X_temp), axis = 1)
                X_test = np.concatenate((X_test, X_test_temp), axis = 1)
            else:
                X = X_temp
                X_test = X_test_temp
                Y = Y_temp
                Y_test = Y_test_temp

        train_acc, test_acc, _ = train_svm([X, X_test, Y, Y_test], c)
        train_acc_list = np.append(train_acc_list,train_acc)
        test_acc_list = np.append(test_acc_list,test_acc)
        comb_list = np.append(comb_list, list(iter))

    return np.array(train_acc_list), np.array(test_acc_list), np.array(comb_list)


def print_svm(acc_train_list, acc_test_list, f_methods, params):
    """Print the results of the SVM grid search

    Parameters
    ----------
        acc_train_list : numpy array
            List of training accuracies
        acc_test_list : numpy array 
            List of test accuracies

    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    max_abs = np.abs(acc_train_list).max()
    plt.imshow(acc_train_list.reshape(5, 15), cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=max_abs)
    for (j, k), label in np.ndenumerate(acc_train_list.reshape(5, 15)):
        label = round(label*100, 1)
        plt.text(k, j, label, ha='center', va='center', fontsize=8)
    labels = [str(iter) for i in range(4) for iter in it.combinations([0,1,2,3], i+1)]
    params = [str(i) for i in params]
    #print(labels)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(params)), params)
    plt.xlabel("Feature combination")
    plt.ylabel("C", rotation=0)
    plt.title("Train accuracy")

    plt.subplot(2, 1, 2)
    max_abs = np.abs(acc_test_list).max()
    plt.imshow(acc_test_list.reshape(5, 15), cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=max_abs)
    for (j, k), label in np.ndenumerate(acc_test_list.reshape(5, 15)):
        label = round(label*100, 1)
        plt.text(k, j, label, ha='center', va='center', fontsize=8)
    labels = [str(iter) for i in range(4) for iter in it.combinations([0,1,2,3], i+1)]
    #print(labels)
    plt.xticks(np.arange(len(labels)), labels , rotation=45)
    params = [str(i) for i in params]
    plt.yticks(np.arange(len(params)), params)
    plt.xlabel("Feature combination")
    plt.ylabel("C", rotation=0)
    plt.title("Test accuracy")

    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)

    # place a text box in upper left in axes coords
    legend = '\n'.join((str(i)+' = '+ e for i, e in enumerate(f_methods)))
    #print(legend)

    plt.text(-10, -2, legend, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig("img/svm.png", dpi=300)
    plt.show()

def grid_search():
    """Perform a grid search for the best combination of features and C parameter

    """
    
    f_methods = [f[:-13] for f in os.listdir('low_features') if f.endswith('_train.txt.gz')]
    f_methods.sort()

    #print(f_methods)

    acc_train_list = np.array([])
    acc_test_list = np.array([])
    comb_list = np.array([])


    #C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
    for param in [1e-1, 1e1, 1e3, 1e5, 1e7]:
        for i in range(len(f_methods)):
            acc_train, acc_test, comb = feature_n_combination(i+1, f_methods, param)
            acc_train_list = np.append(acc_train_list, acc_train)
            acc_test_list = np.append(acc_test_list, acc_test)
            comb_list = np.append(comb_list, comb)
    

    print_svm(acc_train_list, acc_test_list, f_methods, ['1e-1', '1e1', '1e3', '1e5', '1e7'])


if __name__ == "__main__":

    acc_train, acc_test, svm = train_svm(insp.import_features('edge_direction_histogram')[0:4], 1e7)

    print(acc_train)
    print(acc_test)
    print(svm.__class__.__name__)

    test_dataset = LowFeaturesTest(transform = ToTensor(), name = 'edge_direction_histogram')
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)

    cmat = conf_mat(test_loader, svm)
    display_conf_mat(cmat, test_loader.dataset.klass ,'svm')



   
    







