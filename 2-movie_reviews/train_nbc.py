import numpy as np
import matplotlib.pyplot as plt

import extract_features as ef
import validation as val

#BOW training
def _train_nb(X : np.ndarray, Y : np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    m, n = X.shape
    k  = Y.max() + 1
    probs = np.zeros((k, n))
    for i in range(k):
        counts = X[Y == i].sum(0)
        tot = counts.sum()
        probs[i, :] = (counts + 1) / (tot + n)
    
    prior = np.bincount(Y) / m

    w = np.log(probs[1,:] / probs[0,:])
    b = np.log(prior[1]   / prior[0]  )

    return w, b

def _inference_nb(X : np.ndarray, w : np.ndarray, b : np.ndarray) -> int: 
    score = X @ w + b
    return (score > 0).astype(int)

def max_min_weights(w : np.ndarray):
    f = open('vocabulary.txt', encoding='utf-8')
    words = f.read()

    i_w_max = np.argsort(w)
    w_max = np.sort(w)
    print("Most positive words:")

    p_pos = []
    v_pos = []
    for i in range(20):
        p_pos.append(words.split()[i_w_max[-(i+1)]])
        v_pos.append(w_max[-(i+1)])
        print(i+1, '', words.split()[i_w_max[-(i+1)]], '', w_max[-(i+1)])

    print('')
    print("Most negative words:")

    p_neg = []
    v_neg = []
    for i in range(20):
        p_neg.append(words.split()[i_w_max[i]])
        v_neg.append(w_max[i])
        print(i+1, '', words.split()[i_w_max[i]], '', w_max[i])

    
    plt.figure(figsize=(25,8), dpi=80)
    plt.subplot(2, 1, 2)
    plt.hlines(y=0, xmin=0, xmax=20, color='black', alpha=0.4, linewidth=1)
    plt.vlines(x=p_neg, ymin=[0]*20, ymax=v_neg, color='firebrick', alpha=0.7, linewidth=2)
    plt.xlabel("Weights")
    plt.ylabel("Value")
    plt.savefig('img/w_neg.png')

    plt.subplot(2, 1, 1)
    plt.hlines(y=0, xmin=0, xmax=20, color='black', alpha=0.4, linewidth=1)
    plt.vlines(x=p_pos, ymin=[0]*20, ymax=v_pos, color='firebrick', alpha=0.7, linewidth=2)
    plt.xlabel("Weights")
    plt.ylabel("Value")
    plt.savefig('img/w_pos.png')

    plt.show()
    
    f.close()

#------------------- main -------------------#



dim = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
a_val = []
a_train = []

ef.extract_from_dir('train', 2500)
    
data = np.load('train.npy')
X = data[:, :-1]
Y = data[:, -1].astype(int)

w, b = _train_nb(X, Y)

predictions = _inference_nb(X, w, b)
accuracy = (predictions == Y).mean()

np.savez('NBC_model.npz', w = w, b = b)

a_train.append(accuracy)
a_val.append(val.validation('NBC'))

print("Accuracy Validation: {:.6f}%".format(a_val[-1] * 100))
print("Accuracy Train: {:.6f}%".format(a_train[-1] * 100))

#max_min_weights(w)

"""
for i in dim:

    ef.extract_from_dir('train', i)

    data = np.load('train.npy')
    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    w, b = _train_nb(X, Y)

    predictions = _inference_nb(X, w, b)
    accuracy = (predictions == Y).mean()

    np.savez('model.npz', w = w, b = b)

    #print("Accuracy: {:.6f}%".format(accuracy * 100))
    a_train.append(accuracy)
    a_val.append(val.validation('model'))

    print("Accuracy: {:.6f}%".format(a_val[-1] * 100))
    print("Accuracy: {:.6f}%".format(a_train[-1] * 100)) 
"""

#np.savetxt('results.txt', [dim, a_train, a_val], fmt='%f', delimiter=' ', newline='\n', header='dim a_train a_val', footer='', comments='# ', encoding=None)

    
