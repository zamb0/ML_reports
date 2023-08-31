import numpy as np
import matplotlib.pyplot as plt
import os
import cake_classification.image_features as im_f


classes = os.listdir('cake_images/train')
classes = [klass for klass in classes if klass[0] != '.']
classes.sort()
#print(classes)

def features_function(file):
    function_names = [getattr(file, func) for func in dir(file) if not func.startswith('__') and not func.startswith('_') and not func.startswith('np')]
    return function_names

def process_directory(path, function):
    all_features = []
    all_labels = []
    klass_labels = 0

    for klass in classes:
        image_file = os.listdir(path + '/' + klass)
        image_file = [image for image in image_file if image[0] != '.']
        for image_name in image_file:
            image = plt.imread(path + '/' + klass + '/' + image_name)
            image = image / 255.0
            #plt.imshow(image)
            #plt.show()
            #features = im_f.color_histogram(image)
            features = function(image)
            features = features.reshape(-1)
            all_features.append(features)
            all_labels.append(klass_labels)
        klass_labels += 1
    
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


for function in features_function(im_f):
    X, Y = process_directory('cake_images/train', function)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt('mlp_data/'+str(function.__name__)+'_train.txt.gz', data)

    X, Y = process_directory('cake_images/test', function)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt('mlp_data/'+str(function.__name__)+'_test.txt.gz', data)


