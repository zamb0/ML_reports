import numpy as np
import extract_features as ef


def _inference_nb(X : np.ndarray, w : np.ndarray, b : np.ndarray) -> int: 
    score = X @ w + b
    return (score > 0).astype(int)


def _find_worst(X : np.ndarray, Y : np.ndarray, w : np.ndarray, b : np.ndarray) -> int:
    

    score = X @ w + b
    data = [score, (score > 0).astype(int), Y]
    worst = np.sort(np.where(data[1] != data[2], data[0], 0))

    return worst[:5], worst[-5:]
    
    

def validation(model : str) -> float:
    ef.extract_from_dir('validation', None)

    data = np.load('validation.npy')
    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    model = np.load(model.upper()+'_model.npz')
    w, b = model["w"], model["b"]

    predictions = _inference_nb(X, w, b)
    accuracy = (predictions == Y).mean()

    #worst_pos, worst_neg = _find_worst(X, Y, w, b)
    #print("Worst Pos: ", worst_pos)
    #print("Worst Neg: ", worst_neg)

    return accuracy

