import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import os
from mlp import LowFeaturesTrain, LowFeaturesTest
from mlp import MLPNet
from mlp import ToTensor
import mlp
from utility import conf_mat, display_conf_mat, top_miss_classified, plot_miss

import inspect_data as insp

device = torch.device('mps')

if __name__ == "__main__":

    f_methods = [f[:-13] for f in os.listdir('low_features') if f.endswith('_train.txt.gz')]
    f_methods.sort()

    train_dataset = LowFeaturesTrain(transform = ToTensor(), name = f_methods[2])
    test_dataset = LowFeaturesTest(transform = ToTensor(), name = f_methods[2])

    input_size = train_dataset.x.shape[1]
    num_classes = len(np.unique(train_dataset.y))
    batch_size = 32

    print("Input size: ", input_size)
    print("Num classes: ", num_classes)
    print("Batch size: ", batch_size)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    model = MLPNet(input_size, num_classes).to(device)
    weights = torch.load("mlp.pth")
    model.load_state_dict(weights)

    criterion = nn.CrossEntropyLoss()

    print(f'Accuracy: {mlp.accuracy(model, test_loader)} %')

    cmat = conf_mat(test_loader, model)
    display_conf_mat(cmat, test_dataset.klass, 'mlp')



