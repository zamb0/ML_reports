import numpy as np
import pvml
import matplotlib.pyplot as plt
import os


def meanvar_normalization(X, mu=None, std=None):
    if mu is None and std is None:
        mu = X.mean(0)
        std = X.std(0)

    X = (X - mu) / std
    return X, mu, std

def plot_norms():

    data = {}
    for file in os.listdir("norm_mdl"):
        if file.endswith("res.txt"):
            file_path = os.path.join("norm_mdl", file)
            d = np.loadtxt(file_path)
            #print(d)
            data[file[:-8]] = d

    #print(data)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylim(0.7, 1)
    plt.xlim(0, 50)
    for key in data:
        plt.plot(1-data[key][:, 0], label=key)
    plt.legend()
    plt.title("Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(visible=True)
    
    plt.subplot(1, 2, 2)
    plt.ylim(0.7, 1)
    plt.xlim(0, 50)
    for key in data:
        plt.plot(1-data[key][:, 1], label=key)
    plt.legend()
    plt.title("Validation")
    plt.xlabel("Epoch")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig("img/norms.png", dpi=300)
    plt.show()

#https://stackoverflow.com/questions/62008143/finding-highest-values-zone-in-a-2d-matrix-in-python
#modified to have a rectangular zone
#tryed to highlight the zone with the highest values...
def get_heighest_zone(arr, zone_size_x, zone_size_y):
    max_sum = float("-inf")
    row_idx, col_idx = 0, 0
    for row in range(arr.shape[0]-zone_size_x):
        for col in range(arr.shape[1]-zone_size_y):
            curr_sum =  np.sum(arr[row:row+zone_size_x, col:col+zone_size_y])
            if curr_sum > max_sum:
                row_idx, col_idx = row, col
                max_sum = curr_sum

    return row_idx, col_idx

def show_weights(network=pvml.MLP.load("norm_mdl/nothing_mlp.npz")):
    w = network.weights[0]

    max_abs = np.abs(w).max()

    plt.figure(figsize=(10, 5))
    
    for klass in range(35):
        plt.subplot(7, 5, klass + 1)
        plt.imshow(w[:,klass].reshape(20, 80), cmap='seismic', vmin=-max_abs, vmax=max_abs)
        plt.title(words[klass])
        plt.axis('off')

    plt.subplots_adjust(bottom=0, right=0.8, top=0.99, hspace=0, wspace=0.07, left=0.03)
    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)
    plt.savefig("img/weights.png", dpi=300)

    plt.show()

def make_confusion_matrix(prediction, label):
    cmat = np.zeros((35, 35), dtype=int)
    for i in range(len(prediction)):
        cmat[label[i], prediction[i]] += 1

    s = cmat.sum(1, keepdims=True)
    cmat = cmat / s
    return cmat

def T_display_confusion_matrix(cmat):
    print (" "*10, end="")
    for i in range(35):
        print("%4s" % words[i], end="")
    print()
    for i in range(35):
        print("%10s" % words[i], end="")
        for j in range(35):
            val = cmat[i, j]*100
            print("%4.1d" % val, end="")
        print()

def G_display_confusion_matrix(cmat):
    color = 'black'
    plt.figure(figsize=(10, 10))
    plt.imshow(cmat, cmap='Blues')
    plt.xticks(range(35), words, rotation=90)
    plt.yticks(range(35), words)
    for i in range(35):
        for j in range(35):
            val = cmat[i, j]*100
            if val > 45:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=6, color='white')
            else:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=6, color=color)
    plt.tight_layout()
    plt.savefig("img/confusion.png", dpi=300)
    plt.show()

def plot_accuracy():
    data = {}
    cut = -8
    for file in os.listdir("epoch_mdl"):
        if file.endswith("res.txt"):
            file_path = os.path.join("epoch_mdl", file)
            d = np.loadtxt(file_path)
            #print(d)
            data[file[:cut]] = d

    #print(data["1"][:,0])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylim(0.7, 1)
    plt.xlim(0, 40)
    for key in sorted(data.keys(), key=lambda x: int(x)):
        if key == "84291":
            plt.plot(1-data[key][:, 0], label='M')
        else:
            plt.plot(1-data[key][:, 0], label=key)
            #print(data[key][:,0])
    plt.legend(title = "Batch size")
    plt.title("Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(visible=True)
    
    plt.subplot(1, 2, 2)
    plt.ylim(0.7, 1)
    plt.xlim(0, 40)
    for key in sorted(data.keys(), key=lambda x: int(x)):
        if key == "84291":
            plt.plot(1-data[key][:, 0], label='M')
        else:
            plt.plot(1-data[key][:, 1], label=key)
    plt.legend(title = "Batch size")
    plt.title("Validation")
    plt.xlabel("Epoch")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig("img/batch.png", dpi=300)
    plt.show()

def plot_accuracy_L():
    data = {}
    cut = -8
    for file in os.listdir("layer_mdl"):
        if file.endswith("res.txt"):
            file_path = os.path.join("layer_mdl", file)
            d = np.loadtxt(file_path)
            #print(d)
            data[file[:cut]] = d

    #print(data["1"][:,0])
    layers = [[1600, 35], [1600, 1000, 35], [1600, 1000, 500, 35], [1600, 1000, 500, 250, 35], [1600, 1000, 500, 250, 100, 35]]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylim(0, 0.9)
    plt.xlim(0, 60)
    i=0
    for key in sorted(data.keys(), key=lambda x: int(x)):
        plt.plot(1-data[key][:, 0], label=layers[i][1:-1])
        i+=1
            #print(data[key][:,0])
    plt.legend(title = "Hidden Layers")
    plt.title("Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(visible=True)
    
    plt.subplot(1, 2, 2)
    plt.ylim(0, 0.9)
    plt.xlim(0, 60)
    i=0
    for key in sorted(data.keys(), key=lambda x: int(x)):
        plt.plot(1-data[key][:, 1], label=layers[i][1:-1])
        i+=1

    plt.legend(title = "Hidden Layers")
    plt.title("Validation")
    plt.xlabel("Epoch")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig("img/layer.png", dpi=300)
    plt.show()

def find_missclassified(prediction, label):
    miss = []
    for i in range(len(prediction)):
        if prediction[i] != label[i]:
            miss.append(i)
    return miss 

def most_common_hist(miss, label, words):
    miss_label = label[miss]
    count = np.zeros(35)
    for i in range(len(miss_label)):
        count[miss_label[i]] += 1

    top5 = min(np.sort(count)[-5:])
    colors = ['red' if val >= top5 else 'blue' for val in count]

    plt.figure(figsize=(10, 5))
    plt.bar(words, count, edgecolor='black', linewidth=1.2, color=colors)
    plt.xticks(rotation=90)
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.title("Most common missclassified words")
    plt.tight_layout()
    plt.savefig("img/miss.png", dpi=300)
    plt.show()

def plot_each_class(X, Y, words, ok):
    i=0
    plt.figure(figsize=(10, 5))
    for class_label in np.unique(Y):
        i+=1

        r = np.random.randint(10)
        class_index = np.where(Y == class_label)[0][r]
        #class_index = np.where(Y == 21)[0][i]
        w = X[class_index]
        max_abs = np.abs(w).max()
    
        plt.subplot(7, 5, i)
        plt.imshow(w.reshape(20, 80), cmap='seismic',vmin=-max_abs, vmax=max_abs)
        plt.title(f"{words[class_label]}")
        #plt.title(f"{words[21]}")
        plt.axis('off')

    plt.subplots_adjust(bottom=0, right=0.8, top=0.99, hspace=0, wspace=0.07, left=0.03)
    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)
    #plt.colorbar()
    #plt.tight_layout()
    plt.savefig("img/each_"+ok+".png", dpi=300)
    plt.show()

def find_file(miss):

    f_val = open("speech_comands/validation-names.txt").read().split()
    miss_name = [] 

    for index in miss:
        miss_name.append(f_val[index])
        #print(f_val[index])

    return miss_name


words = open("speech_comands/classes.txt").read().split()
data = np.load("speech_comands/test.npz")
Xtest = data["arr_0"]
Ytest = data["arr_1"]

network = pvml.MLP.load("layer_mdl/3_mlp.npz")
mu, std = np.loadtxt("layer_mdl/param.txt")

Xtest, *_ = meanvar_normalization(Xtest, mu, std)

#show_weights()

prediction, logits = network.inference(Xtest)

accuracy = np.mean(prediction == Ytest)
#print(f"Accuracy: {accuracy*100:.2f}")


cmat = make_confusion_matrix(prediction, Ytest)
#T_display_confusion_matrix(cmat)
#G_display_confusion_matrix(cmat)
#plot_norms()
#plot_accuracy()
#plot_accuracy_L()
miss = find_missclassified(prediction, Ytest)
#most_common_hist(miss, Ytest, words)

#plot_each_class(Xtest, Ytest, words, "all")
#plot_each_class(Xtest[np.delete([i for i in range(len(Xtest))],miss)], np.delete(Ytest,miss) , words, "ok")
#plot_each_class(Xtest[miss], Ytest[miss], words, "miss")
miss_file_name = find_file(miss)




