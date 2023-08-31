import numpy as np
import matplotlib.pyplot as plt
import os
import pvml

def G_display_confusion_matrix(cmat, words, layer):
    color = 'black'
    plt.figure(figsize=(10, 10))
    plt.imshow(cmat, cmap='Blues')
    plt.xticks(range(15), words, rotation=90)
    plt.yticks(range(15), words)
    for i in range(15):
        for j in range(15):
            val = cmat[i, j]*100
            if val > 45:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=10, color='white')
            elif val > 13 and i != j:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=10, color='red')
            else:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=10, color=color)
    plt.tight_layout()
    plt.savefig('img/confusion_'+layer+'.png', dpi=300)
    plt.show()

def conf_mat(classes, cnn, layer):
    cmat = np.zeros((len(classes), len(classes)), dtype=float)

    klasses = [klass for klass in os.listdir('cake_classification/images/test') if klass[0] != '.']
    for Klass in klasses:
        Klass_path = 'cake_classification/images/test/' + Klass
        for image_name in os.listdir(Klass_path):
            image = plt.imread(Klass_path + '/' + image_name)
            image = image / 255.0
            label, probs = cnn.inference(image[None, :, :, :])

            cmat[classes.index(Klass), label[0]] += 1
            #print(f"{Klass} {classes[label[0]]} {probs[0][label[0]] * 100:.1f}")

    cmat_p = cmat / cmat.sum(1, keepdims=True)
    G_display_confusion_matrix(cmat_p, classes, layer)
    return np.sum(np.diag(cmat))/np.sum(cmat)

def conf_mat_MLP(classes, Xtest, Ytest, info):
    cmat = np.zeros((len(classes), len(classes)), dtype=float)

    prediction, probs = mlp.inference(Xtest)

    for i in range(len(prediction)):
        cmat[Ytest[i], prediction[i]] += 1
        #print(f"{Klass} {classes[label[0]]} {probs[0][label[0]] * 100:.1f}")

    cmat_p = cmat / cmat.sum(1, keepdims=True)
    G_display_confusion_matrix(cmat_p, classes, info)
    return np.sum(np.diag(cmat))/np.sum(cmat)

def top_miss_classified(cnn):
    miss = []
    
    klasses = [klass for klass in os.listdir('cake_classification/images/test') if klass[0] != '.']
    for Klass in klasses:
        Klass_path = 'cake_classification/images/test/' + Klass
        for image_name in os.listdir(Klass_path):
            image = plt.imread(Klass_path + '/' + image_name)
            image = image / 255.0
            label, probs = cnn.inference(image[None, :, :, :])

            if Klass != classes[label[0]]:
                miss.append([Klass, classes[label[0]], probs[0][label[0]], Klass_path + '/' + image_name])
    
    return sorted(miss, key=lambda x: x[2], reverse=True)[:10]

def top_miss_classified_MLP(mlp):
    miss = []
    path = []
   
    prediction, probs = mlp.inference(X_test)

    #klasses = [klass for klass in os.listdir('cake_classification/images/test') if klass[0] != '.']
    for klass in classes:
        for image_name in os.listdir('cake_classification/images/test/'+klass):
                path.append('cake_classification/images/test/'+klass+'/'+image_name)

    for i in range(len(prediction)):
        if Y_test[i] != prediction[i]:
            miss.append([classes[Y_test[i]], classes[prediction[i]], probs[i][prediction[i]], path[i]])

    

    return sorted(miss, key=lambda x: x[2], reverse=True)[:10]

def make_cnn(cnn, mlp, layer):
    print(mlp.weights[0].shape)

    for lay in range(len(cnn.weights)):
        print(np.shape(cnn.weights[lay]))

    cnn.weights[-1] = mlp.weights[0][None, None, :, :]  # (1, 1, 1024, 15) -> (1024, 15)
    cnn.biases[-1] = mlp.biases[0]

    for lay in range(len(cnn.weights)):
        print(np.shape(cnn.weights[lay]))


    return cnn

def plot_miss(miss, tipe):
    plt.figure(figsize=(12, 4))
    for i in range(len(miss)):
        print(f"{i+1:3} {miss[i][0]:20} {miss[i][1]:20} {miss[i][2] * 100:1f} {miss[i][3]}")

        plt.subplot(2, 5, i+1)
        plt.imshow(plt.imread(miss[i][3]))
        plt.title(f"{miss[i][0]} -> {miss[i][1]}   {miss[i][2] * 100:.1f}", fontsize=8)
        plt.axis('off') 

    plt.tight_layout()
    plt.savefig("img/top_miss_"+tipe+"_"+file[12:-4]+".png", dpi=300)
    plt.show()

def import_features(path):

    data  = np.loadtxt(path+'_train.txt.gz')
    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    data  = np.loadtxt(path+'_test.txt.gz')
    X_test = data[:, :-1]
    Y_test = data[:, -1].astype(int)

    return X, Y, X_test, Y_test



classes = os.listdir('cake_classification/images/test')
classes = [klass for klass in classes if klass[0] != '.']
classes.sort()




files = [file for file in os.listdir('cnn_res') if file[-4:] == '.npz' and file[:4] == 'cake']
print(files)
for file in files:
    #cnn = pvml.CNN.load('cake_classification/pvmlnet.npz')

    X, Y, X_test, Y_test = import_features(os.path.join('cnn_data', 'pvmlnet'+file[12:-4]))

    mlp = pvml.MLP.load(os.path.join('cnn_res', file))

    #cnn = make_cnn(cnn, mlp, int(file[12:-4]))
    #cnn.save(os.path.join('cnn_res', 'cnn_'+file[12:-4]+'.npz'))

    #imagepath = 'cake_classification/images/test/apple_pie/2856273.jpg'
    #image = plt.imread(imagepath)
    #image = image / 255.0

    #label, probs = cnn.inference(image[None, :, :, :])
    #print(probs.shape)

    #print(classes)

    #indices = (-probs[0]).argsort()
    #print(indices.shape)
    #for i in range(5):
    #    index = indices[i]
    #    #print(index)
    #    print(f"{i+1} {classes[index]:10} {probs[0][index] * 100:.1f}")

    #plt.imshow(image)
    #plt.show()

    #acc = conf_mat(classes, cnn, file[12:-4])
    #print(f"Accuracy: {acc*100:.1f}%")
    #acc_mlp = conf_mat_MLP(classes, X_test, Y_test, file[12:-4])
    #print(f"Accuracy: {acc_mlp*100:.1f}%")

    #miss_cnn = top_miss_classified(cnn)
    #miss_mlp = top_miss_classified_MLP(mlp)

    #print formatted miss

    #plot_miss(miss_cnn, 'cnn')
    #plot_miss(miss_mlp, 'mlp')

    







       





