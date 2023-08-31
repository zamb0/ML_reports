import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm, trange


import titanic_test as tt

def logreg_inference(X, w, b):
    logit = X @ w + b
    probability = 1 / (1 + np.exp(-logit))

    return probability

def cross_entropy_loss(p, y):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

def logreg_train(X, Y, lambda_, lr, num_iters):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_l = []
    loss_lt = []
    accuracy_l = []
    accuracy_lt = []

    t = trange(num_iters, desc='Train', leave=True)
    for i in t:
        p = logreg_inference(X, w, b)
        grad_w = X.T @ (p - Y) / m + lambda_ * w
        grad_b = (p - Y).mean()

        w -= lr * grad_w
        b -= lr * grad_b

        if i % 10000 == 0:
            predictions = (p > 0.5)
            accuracy = (predictions == Y).mean()
            loss = cross_entropy_loss(p, Y)
            t.set_postfix({'accuracy': accuracy, 'loss': loss})
            
            loss_l.append(loss)
            accuracy_l.append(accuracy)
            
            loss_t, accl_t = tt.test(w, b)
            accuracy_lt.append(accl_t)
            loss_lt.append(loss_t)


    return w, b, loss_l, accuracy_l, loss_lt, accuracy_lt


def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

X, Y = load_file("titanic-data/titanic-train.txt")    

#print(X.shape, Y.shape)  

w, b, losses, accuracies, loss_test, accuracies_test = logreg_train(X, Y, 0.0015, 0.001, 3000000)

# save the model
np.savez("titanic-model.npz", w = w, b = b)

#print("w:", w)
#print("b:", b)

#plot weights
plt.figure()
plt.hlines(y=0, xmin=0, xmax=6, color='black', alpha=0.4, linewidth=1)
plt.vlines(x=['Pclass','Sex','Age','S/S Aboard', 'P/C Aboard', 'Fare'], ymin=[0, 0, 0, 0, 0, 0], ymax=w, color='firebrick', alpha=0.7, linewidth=2)
plt.xlabel("Weights")
plt.ylabel("Value")
#plt.savefig('img/weights.png', dpi=300)
#plt.show()

#loss vs accuracy
plt.figure()
plt.plot(losses)
plt.plot(accuracies)
plt.axvline(x = 275, color = 'r', linestyle = '--')
plt.ylabel("Loss/Accuracy")
plt.xlabel("Iterations (x10000)")
plt.legend(["Loss", "Accuracy"])
#plt.savefig('img/lvsa.png', dpi=300)
#plt.show()

#loss
plt.figure()
plt.plot(losses)
plt.xlabel("Iterations (x10000)")
plt.axvline(x = 275, color = 'r', linestyle = '--')
plt.ylabel("Loss")
plt.legend(["Loss", "Convergence"])
#plt.savefig('img/loss.png', dpi=300)
#plt.show()

#accuracy
plt.figure()
plt.plot(accuracies)
plt.axvline(x = 275, color = 'r', linestyle = '--')
plt.xlabel("Iterations (x10000)")
plt.legend(["Accuracy","Convergence"])
#plt.savefig('img/accuracy.png', dpi=300)
#plt.show()

#accuracy test vs accuracy train
plt.figure()
plt.plot(accuracies_test)
plt.plot(accuracies)
plt.axvline(x = 275, color = 'r', linestyle = '--')
plt.ylabel("Accuracy test vs Accuracy train")
plt.xlabel("Iterations (x10000)")
plt.legend(["Test", "Train", "Convergence"])
#plt.savefig('img/testvstrain.png', dpi=300)
#plt.show()

#accuracy test vs accuracy train
plt.figure()
plt.plot(loss_test)
plt.plot(losses)
plt.axvline(x = 275, color = 'r', linestyle = '--')
plt.ylabel("Loss test vs Loss train")
plt.xlabel("Iterations (x10000)")
plt.legend(["Test", "Train", "Convergence"])
#plt.savefig('img/losstestvstrain.png', dpi=300)
#plt.show()


#2.3    [-1.2743143   3.00419274 -0.05761915 -0.40670381 -0.15041257 -0.00464388]
#       the second wheigh is the most important
#2.4    woman from the first class | man from the third class  

my_p = logreg_inference([3, 0, 23, 0, 0, 200], w, b)
#print("My probability:", my_p*100, "%")

plt.figure()
#not very informative
#plt.scatter(X[:, 0], X[:, 1], c = Y)
#i introduce noise to the data
Xn = X + np.random.normal(0, 0.2, X.shape)
plt.scatter(Xn[:, 0], Xn[:, 1], c = Y)
plt.xlabel("Class")
plt.ylabel("Sex")
plt.yticks(X[:,1], ['Female' if x == 1 else 'Male' for x in X[:, 1]])
plt.xticks(X[:,0], ['First' if x == 1 else 'Second' if x == 2 else 'Third' for x in X[:, 0]])
plt.legend(["Died"])
#plt.savefig('img/scatter.png', dpi=300)
#plt.show()

#3
#print("Evaluating the model on the test set ...")
#tt.test(w, b)

#clear the command line
os.system('cls' if os.name == 'nt' else 'clear')

#print the results formatted
print("Train:    Accuracy {:4}   Loss {:4}".format(accuracies[-1], losses[-1]))
print("")
print("Test:     Accuracy {:4}   Loss {:4}".format(accuracies_test[-1], loss_test[-1]))
print("")


