import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as nnF
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

device = torch.device('mps')

def conf_mat(test_loader, model):
    """calculate confusion matrix

    Parameters
    ----------

        tesr_loader : DataLoader
            test data loader
        model : torch.nn.Module or sklearn.svm.SVC
            model to be tested

    Returns
    -------
        cf_matrix : numpy.ndarray   
            confusion matrix
    """
    y_pred = []
    y_true = []

    model.to(device)

    for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if model.__class__.__name__ == 'SVC':
                output = model.predict_proba(inputs.cpu().numpy())
                output = torch.from_numpy(output)
                output = torch.max(output, 1)[1].data.cpu().numpy()
            else:
                output = model(inputs)
                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

            y_pred.extend(output)
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix / np.sum(cf_matrix, axis=1)[:, None]

    return cf_matrix


def display_conf_mat(cmat, words, layer):
    """display confusion matrix

    Parameters
    ----------
        cmat : numpy.ndarray
            confusion matrix
        words : list
            list of class names
        layer : str
            name of the layer
    """
    
    color = 'black'
    plt.figure(figsize=(10, 10))
    plt.imshow(cmat, cmap='Blues')
    #plt.xticks(np.arange(len(words)), words, rotation=90)

    plt.xticks(np.arange(len(words)), ['' for i in range(len(words))], rotation=90)
    img = [plt.imread(os.path.join("road-signs", klass, "train", os.listdir(os.path.join("road-signs", klass, "train"))[0])) for klass in words]
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
                        box_alignment=(0.5, 31)
                        )
        ax.add_artist(ab)


    #plt.yticks(np.arange(len(words)), words)
    plt.yticks(np.arange(len(words)), ['' for i in range(len(words))], rotation=0)
    img = [plt.imread(os.path.join("road-signs", klass, "train", os.listdir(os.path.join("road-signs", klass, "train"))[0])) for klass in words]
    #print(np.array(img).shape)
    ax = plt.gca()
    tick_labels = ax.yaxis.get_ticklabels()
    for i,im in enumerate(img):
        #print(i)
        ib = OffsetImage(im, zoom=.1)
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=(2, 0.5)
                        )
        ax.add_artist(ab)




    for i in range(len(words)):
        for j in range(len(words)):
            val = cmat[i, j]*100
            if val > 45:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=8, color='white')
            elif val > 13 and i != j:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=8, color='red')
            else:
                plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=8, color=color)
    plt.tight_layout()
    plt.savefig('img/confusion_'+layer+'.png', dpi=300)
    plt.show()


def top_miss_classified(model, data_loader):

    """find top missclassified images

    Parameters
    ----------
        model : torch.nn.Module
            model to be tested
        data_loader : DataLoader
            test data loader
            
    Returns
    -------
        miss : list
            list of missclassified images
        img : list
            list of missclassified images
    """

    miss = []
    img = []

    classes = data_loader.dataset.klass
   
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            prob = nnF.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(predicted)):
                if labels[i] != predicted[i]:
                    miss.append([classes[labels[i]], classes[predicted[i]], prob[i][predicted[i]]])
                    img.append(images[i])

    return (list(t) for t in zip(*sorted(zip(miss, img), key=lambda x: x[0][2], reverse=True)))


def plot_miss(miss, miss_img, tipe, n=10):
    """plot missclassified images

    Parameters
    ----------
        miss : list
            list of missclassified images
        miss_img : list
            list of missclassified images
        tipe : str
            type of the model
        n : int
            number of images to be plotted
    """
    
    plt.figure(figsize=(12, 5))

    for i in range(len(miss)):
        if i == n:
            break

        print(f"{i+1:3} {miss[i][0]:20} {miss[i][1]:20} {100*miss[i][2]:1f}")

        plt.subplot(2, n//2, i+1)
        if tipe == 'rgb':
            plt.imshow(miss_img[i].cpu().numpy().transpose(1, 2, 0))
        else:
            plt.imshow(miss_img[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.title(f"{miss[i][0]} -> {miss[i][1]}  {100*miss[i][2]:.1f}", fontsize=6)
        plt.axis('off') 

    plt.tight_layout()
    plt.savefig("img/top_miss_"+tipe+".png", dpi=300)
    plt.show()