import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def read_images(path):
    """Read images from a directory.

    Parameters
    ----------
    path : str
        path to the directory.

    Returns
    -------
    ndarray, shape (N, m, n, 3)
        images in RGB format.

    """

    imgs = []

    for img in os.listdir(path):
        if img.endswith(".jpg"):
            #print(img)
            img = plt.imread(os.path.join(path, img))
            #print(img.shape)
            #plt.imshow(img)
            #plt.show()
            imgs.append(img)

    return np.array(imgs)

def make_dataset(path) :
    """Make a dataset from a directory.

    Parameters
    ----------
    path : str
        path to the directory.

    Returns
    -------
    ndarray, shape (N, m, n, 3)
        Train images in RGB format.
    
    ndarray, shape (N,)
        Train labels.

    ndarray, shape (N, m, n, 3)
        Test images in RGB format.
    
    ndarray, shape (N,)
        Test labels.

    list, shape (k,)
        List of classes.

    """

    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []

    klasses = [klass for klass in os.listdir(path) if klass[0] != '.']
    klasses.sort()
    for klass in klasses:
        for type in os.listdir(os.path.join(path, klass)):
            if type == "train":
                for img in os.listdir(os.path.join(path, klass, type)):
                    if img.endswith(".jpg"):
                        img = plt.imread(os.path.join(path, klass, type, img))
                        img = img / 255.0
                        train_imgs.append(img)
                        train_labels.append(klasses.index(klass))
            elif type == "test":
                for img in os.listdir(os.path.join(path, klass, type)):
                    if img.endswith(".jpg"):
                        img = plt.imread(os.path.join(path, klass, type, img))
                        img = img / 255.0
                        test_imgs.append(img)
                        test_labels.append(klasses.index(klass))

    return np.array(train_imgs), np.array(train_labels), np.array(test_imgs), np.array(test_labels), np.array(klasses)

def class_plot(dataset):
    """Plot images from each class.

    Parameters
    ----------
    dataset : tuple of ndarrays
        dataset returned by make_dataset.

    """

    train_imgs, train_labels, _, _, klasses = dataset

    plt.figure(figsize=(10, 10))
    i=1
    for klass in klasses:
        img = train_imgs[train_labels == list(klasses).index(klass)][0]
        plt.subplot(5, 4, i)
        plt.imshow(img)
        plt.title(klass)
        plt.axis("off")
        i += 1

    plt.tight_layout()
    plt.savefig("img/classes.png", dpi=300)
    plt.show()

def import_features(names):
    """Import features from files.

    Parameters
    ----------
    names : list of str
        names of the features to import.

    Returns
    -------
    ndarray, shape (N, m)
        Train features.

    ndarray, shape (M, m)
        Test features.

    ndarray, shape (N,)
        Train labels.

    ndarray, shape (M,)
        Test labels.

    """

    train_features = np.array([])
    test_features = np.array([])
    train_labels = np.array([])
    test_labels = np.array([])

    if type(names) == str:
        names = [names]

    if names == None:
        names = [f[:-13] for f in os.listdir('low_features') if f.endswith('_train.txt.gz')]
        names.sort()

    print(names)
    for file in os.listdir('low_features'):
        for name in names:
            if file.endswith('_train.txt.gz') and ((file.startswith(name))):
                #print(file)
                data = np.loadtxt('low_features/'+file)
                X = data[:, :-1]
                Y = data[:, -1].astype(int)
                if train_features.size != 0:
                    train_features = np.concatenate((train_features, X), axis = 1)
                else:
                    train_features = X
                    train_labels = Y

            elif file.endswith('_test.txt.gz') and ((file.startswith(name))):
                #print(file)
                data = np.loadtxt('low_features/'+file)
                X = data[:, :-1]
                Y = data[:, -1].astype(int)
                if test_features.size != 0:
                    test_features = np.concatenate((test_features, X), axis = 1)
                else:
                    test_features = X
                    test_labels = Y
            else:
                continue

    klasses = make_dataset("road-signs")[4]

    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels), np.array(klasses)


def print_hist(dataset):
    """Print histogram of the dataset.

    Parameters
    ----------
    dataset : tuple of ndarrays
        dataset returned by make_dataset.

    """
    
    plt.figure(figsize=(10, 6))
    bins = np.arange(21)-0.5
    plt.hist(dataset[1], bins = bins, rwidth = 0.8)
    #plt.xticks(range(20), dataset[4], rotation=90, fontsize=8)
    plt.xticks(range(20), ['' for i in range(20)], rotation=90, fontsize=8)
    img = [plt.imread(os.path.join("road-signs", klass, "train", os.listdir(os.path.join("road-signs", klass, "train"))[0])) for klass in dataset[4]]
    #print(np.array(img).shape)
    ax = plt.gca()
    tick_labels = ax.xaxis.get_ticklabels()
    for i,im in enumerate(img):
        #print(i)
        ib = OffsetImage(im, zoom=.1)
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=(0.5, 1.2)
                        )
        ax.add_artist(ab)
    plt.title("Train dataset")
    #plt.xlabel("Classes")
    plt.ylabel("Number of images")
    plt.subplots_adjust(bottom=0.33)
    plt.savefig("img/dataset_hist.png", dpi=300)
    plt.show()

    
if __name__ == "__main__":
    dataset = make_dataset("road-signs")

    names = [f[:-13] for f in os.listdir('low_features') if f.endswith('_train.txt.gz')]
    names.sort()
    print(names)

    #print(type(dataset[0]))
    #print(type(dataset[1]))
    #print(type(dataset[2]))
    #print(type(dataset[3]))
    #print(type(dataset[4]))

    print(dataset[0].shape)
    print(dataset[1].shape)
    print(dataset[2].shape)
    print(dataset[3].shape)
    print(dataset[4].shape)

    #for i in np.unique(dataset[1]):
    #    print(i, np.sum(dataset[1] == i))

    #rnd = np.random.randint(0, dataset[0].shape[0])
    #plt.imshow(dataset[0][rnd])
    #plt.title(dataset[4][dataset[1][rnd]])
    #plt.show()

    #class_plot(dataset)

