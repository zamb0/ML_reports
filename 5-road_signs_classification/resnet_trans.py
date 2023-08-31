import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from cnn import SignsTrain, SignsTest, ToTensor, ToEraseBlock, ToGaussianNoise
from tqdm import tqdm


device = torch.device("mps")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """train model  

    Parameters
    ----------
        model : torch.nn.Module
            model to be trained
        criterion : torch.nn.modules.loss
            loss function
        optimizer : torch.optim 
            optimizer
        scheduler : torch.optim.lr_scheduler
            learning rate scheduler
        num_epochs : int
            number of epochs

    Returns
    -------
        model : torch.nn.Module
            trained model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    composed_aug = torchvision.transforms.Compose([
						ToTensor(), 
						ToEraseBlock()])
		
    composed_aug2 = torchvision.transforms.Compose([
                        ToTensor(), 
                        ToGaussianNoise()])
    
    composed = torchvision.transforms.Compose([
                        ToTensor()])

    train_dataset = torch.utils.data.ConcatDataset([SignsTrain(transform = composed), SignsTrain(transform = composed_aug), SignsTrain(transform = composed_aug2)])
    test_dataset = SignsTest(transform = ToTensor())


    dataset = { 'train': train_dataset, 
                  'val': test_dataset }
    
    dataloaders = { 'train': DataLoader(dataset = dataset['train'], batch_size = 64, shuffle = True), 
                      'val': DataLoader(dataset = dataset['val'], batch_size = 64, shuffle = True) }
    
    dataset_sizes = { 'train': len(dataset['train']), 
                        'val': len(dataset['val']) }

    class_names = dataset['val'].klass


    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 20)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.01)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=20)

    PATH = './resnet_trans.pth'
    torch.save(model_conv.state_dict(), PATH)
