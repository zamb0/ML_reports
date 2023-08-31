import numpy as np

def test(wt, bt):
    data = np.load("titanic-model.npz")
    w, b = data["w"], data["b"]
    #print(w, b)

    def load_file(filename):
        data = np.loadtxt(filename)
        X = data[:, :-1]
        Y = data[:, -1]
        return X, Y

    def logreg_inference(X, w, b):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        return p

    X, Y = load_file("titanic-data/titanic-test.txt")

    P = logreg_inference(X, w, b)
    p = logreg_inference(X, wt, bt)

    accuracy_t = (Y == (p > 0.5)).mean()
    loss_t = (-Y * np.log(p) - (1 - Y) * np.log(1 - p)).mean()
   
    return loss_t, accuracy_t