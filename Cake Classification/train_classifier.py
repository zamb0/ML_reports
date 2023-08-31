import numpy as np
import matplotlib.pyplot as plt
import os
import pvml
import itertools as it

classes = os.listdir('cake_classification/images/train')
classes = [klass for klass in classes if klass[0] != '.']
classes.sort()

def import_features(path):

    data  = np.loadtxt(path+'_train.txt.gz')
    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    data  = np.loadtxt(path+'_test.txt.gz')
    X_test = data[:, :-1]
    Y_test = data[:, -1].astype(int)

    return X, Y, X_test, Y_test

#print([list(iter) for iter in iter])

def feature_2_combination(r=2, features = ['color_histogram', 'cooccurrence_matrix', 'edge_direction_histogram', 'rgb_cooccurrence_matrix']):
    iter = it.combinations(features, r)
    for iter in iter:

        #Xcolor, Ycolor, Xcolor_test, Ycolor_test = import_features('data/color_histogram')

        #Xcoo, Ycoo, Xcoo_test, Ycoo_test = import_features('data/cooccurrence_matrix')

        #Xedge, Yedge, Xedge_test, Yedge_test = import_features('data/edge_direction_histogram')

        #Xrgb, Yrgb, Xrgb_test, Yrgb_test = import_features('data/rgb_cooccurrence_matrix')


        #X = np.concatenate([Xcolor, Xcoo, Xedge, Xrgb], axis=1)
        #X_test = np.concatenate([Xcolor_test, Xcoo_test, Xedge_test, Xrgb_test], axis=1)
        #Y = Ycolor
        #Y_test = Ycolor_test

        X, Y, X_test, Y_test = import_features('data/'+str(iter[0]))
        X2, Y2, X2_test, Y2_test = import_features('data/'+str(iter[1]))

        X = np.concatenate([X, X2], axis=1)
        X_test = np.concatenate([X_test, X2_test], axis=1)
        Y = Y
        Y_test = Y_test

        input_size = X.shape[1]
        output_size = len(np.unique(Y))
        mlp = pvml.MLP([input_size, output_size])

        #print(X.shape)
        #print(Y.shape)
        #print(input_size)
        #print(output_size)

        M = X.shape[0]
        tr_acc = []
        te_acc = []

        for epoch in range(5000):
            mlp.train(X, Y, lr=0.0001, batch=50, steps=M//50)
            if epoch % 100 == 0:
                prediction, probs = mlp.inference(X)
                train_accuracy = np.mean(prediction == Y)
                tr_acc.append(train_accuracy)

                prediction, probs = mlp.inference(X_test)
                test_accuracy = np.mean(prediction == Y_test)
                te_acc.append(test_accuracy)

                print(f"{epoch} {train_accuracy * 100:.1f} {test_accuracy * 100:.1f}")
                plt.plot(tr_acc, label='train', color='red')
                plt.plot(te_acc, label='test', color='blue')
                plt.legend()
                plt.draw()
                plt.pause(0.01)
                plt.ylim(0, 0.3)
                plt.clf()
                
        plt.plot(tr_acc, label='train', color='red')
        plt.plot(te_acc, label='test', color='blue')
        plt.legend()
        plt.ylim(0, 0.3)
        plt.grid(True)
        #plt.show()
        #plt.close()
        plt.savefig('img/mlp_'+str(list(iter)[0])+'_'+str(list(iter)[1])+'.png', dpi=300)

        np.savetxt('mlp_res/acc_'+str(list(iter)[0])+'_'+str(list(iter)[1])+'.txt', [tr_acc, te_acc])

        mlp.save('mlp_res/cake_'+str(list(iter)[0])+'_'+str(list(iter)[1])+'.npz')


def feature_all():
    Xcolor, Ycolor, Xcolor_test, Ycolor_test = import_features('data/color_histogram')

    Xcoo, Ycoo, Xcoo_test, Ycoo_test = import_features('data/cooccurrence_matrix')

    Xedge, Yedge, Xedge_test, Yedge_test = import_features('data/edge_direction_histogram')

    Xrgb, Yrgb, Xrgb_test, Yrgb_test = import_features('data/rgb_cooccurrence_matrix')


    X = np.concatenate([Xcolor, Xcoo, Xedge, Xrgb], axis=1)
    X_test = np.concatenate([Xcolor_test, Xcoo_test, Xedge_test, Xrgb_test], axis=1)
    Y = Ycolor
    Y_test = Ycolor_test

    input_size = X.shape[1]
    output_size = len(np.unique(Y))
    mlp = pvml.MLP([input_size, output_size])

    #print(X.shape)
    #print(Y.shape)
    #print(input_size)
    #print(output_size)

    M = X.shape[0]
    tr_acc = []
    te_acc = []

    for epoch in range(5000):
        mlp.train(X, Y, lr=0.0001, batch=50, steps=M//50)
        if epoch % 100 == 0:
            prediction, probs = mlp.inference(X)
            train_accuracy = np.mean(prediction == Y)
            tr_acc.append(train_accuracy)

            prediction, probs = mlp.inference(X_test)
            test_accuracy = np.mean(prediction == Y_test)
            te_acc.append(test_accuracy)

            print(f"{epoch} {train_accuracy * 100:.1f} {test_accuracy * 100:.1f}")
            plt.plot(tr_acc, label='train', color='red')
            plt.plot(te_acc, label='test', color='blue')
            plt.legend()
            plt.ylim(0, 0.3)
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            
    plt.plot(tr_acc, label='train', color='red')
    plt.plot(te_acc, label='test', color='blue')
    plt.legend()
    plt.ylim(0, 0.3)
    plt.grid(True)
    #plt.show()
    #plt.close()
    plt.savefig('img/mlp_all.png', dpi=300)

    np.savetxt('mlp_res/acc_all.txt', [tr_acc, te_acc])

    mlp.save('mlp_res/cake_all.npz')

def feature_neural(layer):

    X, Y, X_test, Y_test = import_features('cnn_data/pvmlnet'+str(layer))

    input_size = X.shape[1]
    output_size = len(np.unique(Y))
    mlp = pvml.MLP([input_size, output_size])

    #print(X.shape)
    #print(Y.shape)
    #print(input_size)
    #print(output_size)

    M = X.shape[0]
    tr_acc = []
    te_acc = []

    for epoch in range(1000):
        mlp.train(X, Y, lr=0.0001, batch=50, steps=M//50)
        if epoch % 10 == 0:
            prediction, probs = mlp.inference(X)
            train_accuracy = np.mean(prediction == Y)
            tr_acc.append(train_accuracy)

            prediction, probs = mlp.inference(X_test)
            test_accuracy = np.mean(prediction == Y_test)
            te_acc.append(test_accuracy)

            print(f"{epoch} {train_accuracy * 100:.1f} {test_accuracy * 100:.1f}")
            #plt.plot(tr_acc, label='train', color='red')
            #plt.plot(te_acc, label='test', color='blue')
            #plt.legend()
            #plt.ylim(0, 0.3)
            #plt.draw()
            #plt.pause(0.01)
            #plt.clf()
            
    plt.plot(tr_acc, label='train', color='red')
    plt.plot(te_acc, label='test', color='blue')
    plt.legend()
    #plt.ylim(0, 0.3)
    plt.grid()
    #plt.close()
    plt.savefig('img/cake_neural_'+str(layer)+'.png', dpi=300)
    plt.show()

    np.savetxt('cnn_res/acc_neural_'+str(layer)+'.txt', [tr_acc, te_acc])

    mlp.save('cnn_res/cake_neural_'+str(layer)+'.npz')


def fine_tuning(cnn):

    if 'cnn_fine_tuning.npz' in os.listdir('cnn_res'):
        print('CNN finded in cnn_res folder')
        cnn = pvml.CNN.load('cnn_res/cnn_fine_tuning.npz')
        tr_acc, te_acc = np.loadtxt('cnn_res/acc_fine_tuning.txt')
        tr_acc = list(tr_acc)
        te_acc = list(te_acc)
    else:
        print('CNN not finded in cnn_res folder')
        cnn = pvml.CNN.load(cnn)
        tr_acc = []
        te_acc = []

    X = []
    Y = []
    path = 'cake_classification/images/train'
    i=0
    for klass in classes:
        image_file = os.listdir(path + '/' + klass)
        image_file = [image for image in image_file if image[0] != '.']
        for image_name in image_file:
            image = plt.imread(path + '/' + klass + '/' + image_name)
            image = image / 255.0
            X.append(image)
            Y.append(i)

        i+=1
    
    X_test = []
    Y_test = []
    path = 'cake_classification/images/test'
    i=0
    for klass in classes:
        image_file = os.listdir(path + '/' + klass)
        image_file = [image for image in image_file if image[0] != '.']
        for image_name in image_file:
            image = plt.imread(path + '/' + klass + '/' + image_name)
            image = image / 255.0
            X_test.append(image)
            Y_test.append(i)
        i+=1
    
    X = np.stack(X, 0)
    Y = np.array(Y)
    X_test = np.stack(X_test, 0)
    Y_test = np.array(Y_test)


    print(X.shape)
    print(Y.shape)

    M = len(X)
    for epoch in range(20):
        cnn.train(X, Y, lr=0.00001, batch=50, steps=M//50)
    
        prediction, probs = cnn.inference(X)
        train_accuracy = np.mean(prediction == Y)
        tr_acc.append(train_accuracy)

        prediction, probs = cnn.inference(X_test)
        test_accuracy = np.mean(prediction == Y_test)
        te_acc.append(test_accuracy)

        print(f"{epoch} {train_accuracy * 100:.1f} {test_accuracy * 100:.1f}")

    cnn.save('cnn_res/cnn_fine_tuning.npz')
    np.savetxt('cnn_res/acc_fine_tuning.txt', [tr_acc, te_acc])    



#feature_2_combination()
#feature_all()

#for layer in range(3, 7):
#    feature_neural(layer)

fine_tuning('cnn_res/cnn_3.npz')