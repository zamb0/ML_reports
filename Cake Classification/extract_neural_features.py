import numpy as np
import matplotlib.pyplot as plt
import os
import pvml 


classes = os.listdir('cake_classification/images/train')
classes = [klass for klass in classes if klass[0] != '.']
classes.sort()
#print(classes)

def external_neural_features(image, cnn, layer):
    activation = cnn.forward(image[None, :, :, :])
    features = activation[layer]
    print(features.shape)
    if features.shape[1] != 1 and features.shape[2] != 1:
        f1, f2 = features[:, :2, :, :], features[:, 2:, :, :]
        features = np.concatenate([f1, f2], axis=3)
        features = np.mean(features, axis=(1, 2))
        print(features.shape)
        #features = np.concatenate([features.reshape(features.shape[1])]*(1024//features.shape[1]), axis=0)

    features = features.reshape(1024)
    return features

def process_directory(path, cnn, layer):
    all_features = []
    all_labels = []
    klass_labels = 0

    for klass in classes:
        image_file = os.listdir(path + '/' + klass)
        image_file = [image for image in image_file if image[0] != '.']
        for image_name in image_file:
            image = plt.imread(path + '/' + klass + '/' + image_name)
            image = image / 255.0

            features = external_neural_features(image, cnn, layer)

            print(features.shape)
            all_features.append(features)
            all_labels.append(klass_labels)
        klass_labels += 1
        print(klass_labels)
    
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


cnn = pvml.CNN.load('cake_classification/pvmlnet.npz')

for layer in range(3, 7):

    X, Y = process_directory('cake_classification/images/train', cnn, -layer)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt('cnn_data/'+'pvmlnet'+str(layer)+'_train.txt.gz', data)

    X, Y = process_directory('cake_classification/images/test', cnn, -layer)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt('cnn_data/'+'pvmlnet'+str(layer)+'_test.txt.gz', data)


