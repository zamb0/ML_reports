import torch
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from cnn import SignsTrain, SignsTest, ToTensor
from utility import conf_mat, display_conf_mat, top_miss_classified, plot_miss
import torch.nn as nn

device = torch.device('mps')


resnet_trans = models.resnet18().to(device)
num_ftrs = resnet_trans.fc.in_features
resnet_trans.fc = nn.Linear(num_ftrs, 20)

resnet_trans.load_state_dict(torch.load('./resnet_trans.pth'))


resnet_fine = models.resnet18().to(device)
num_ftrs = resnet_fine.fc.in_features
resnet_fine.fc = nn.Linear(num_ftrs, 20)

resnet_fine.load_state_dict(torch.load('./resnet_fine.pth'))


test_loader = DataLoader(dataset = SignsTest(transform = ToTensor()), batch_size = 32, shuffle = True)


#Transfer
cmat = conf_mat(test_loader, resnet_trans)
display_conf_mat(cmat, test_loader.dataset.klass, 'resnet_trans')

miss, miss_img = top_miss_classified(resnet_trans, test_loader)
plot_miss(miss, miss_img, 'resnet_trans')


#Fine
cmat = conf_mat(test_loader, resnet_fine)
display_conf_mat(cmat, test_loader.dataset.klass, 'resnet_fine')

miss, miss_img = top_miss_classified(resnet_fine, test_loader)
plot_miss(miss, miss_img, 'resnet_fine')

