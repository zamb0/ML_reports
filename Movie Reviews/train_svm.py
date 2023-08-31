import numpy as np

import extract_features as ef
import validation as val

def _svm_train(X, Y, lambda_=0, lr=1e-3, steps=1000):
    m, n = X.shape
    w = np.zeros(n)
    b=0
    for step in range(steps):
        z=X@w+b
        hinge_diff = -Y * (z < 1) + (1 - Y) * (z > -1) 
        grad_w = (hinge_diff @ X) / m + lambda_ * w 
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b 
    
    return w, b

def _svm_inference(X, w, b): 
    z=X@w+b
    labels = (z > 0).astype(int) 
    return labels


iter = [1000, 2500, 5000, 7500]
lr = [1e-3, 1e-2, 1e-1, 1]
lamb = [0, 1e-3, 1e-2, 1e-1]
a_val = []
a_train = []
a_iter = []
a_lr = []
a_lamb = []

ef.extract_from_dir('train', 2500)
                
data = np.load('train.npy')
X = data[:, :-1]
Y = data[:, -1].astype(int)

w, b = _svm_train(X, Y, lambda_=0, lr=0.1, steps=2500)

np.savez('SVM_model.npz', w = w, b = b)

predictions = _svm_inference(X, w, b)
accuracy = (predictions == Y).mean()

a_train.append(accuracy)
a_val.append(val.validation('SVM'))

print("Accuracy val: {:.6f}%".format(a_val[-1] * 100))
print("Accuracy train: {:.6f}%".format(a_train[-1] * 100)) 

'''
for i in iter:
    for j in lr:
        for k in lamb:
            
            w, b = _svm_train(X, Y, lambda_=k, lr=j, steps=i)

            np.savez('model_svm.npz', w = w, b = b)

            predictions = _svm_inference(X, w, b)
            accuracy = (predictions == Y).mean()

            a_train.append(accuracy)
            a_val.append(val.validation('model_svm'))
            a_iter.append(i)
            a_lr.append(j)
            a_lamb.append(k)

            print("Iter: {}, lr {}, lamb {}".format(i, j, k))
            print("Accuracy val: {:.6f}%".format(a_val[-1] * 100))
            print("Accuracy train: {:.6f}%".format(a_train[-1] * 100)) 
'''

#np.savetxt('results_svm.txt', [a_iter, a_lr, a_lamb, a_train, a_val], fmt='%f', delimiter=' ', newline='\n', header='iter lr lambda a_train a_val', footer='', comments='# ', encoding=None)


