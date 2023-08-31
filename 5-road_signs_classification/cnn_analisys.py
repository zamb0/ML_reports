import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from cnn import ColorConvNet, GrayConvNet
from cnn import SignsTest
from cnn import ToTensor, ToGray
import cnn
from utility import conf_mat, display_conf_mat, top_miss_classified, plot_miss

device = torch.device('mps')

rgb_model = ColorConvNet().to(device)
gray_model = GrayConvNet().to(device)
rgb_aug_model = ColorConvNet().to(device)
gray_aug_model = GrayConvNet().to(device)


weights = torch.load("rgb_cnn.pth")
rgb_model.load_state_dict(weights)

weights = torch.load("gray_cnn.pth")
gray_model.load_state_dict(weights)

weights = torch.load("rgb_aug_cnn.pth")
rgb_aug_model.load_state_dict(weights)

weights = torch.load("gray_aug_cnn.pth")
gray_aug_model.load_state_dict(weights)


#RGB
test_dataset = SignsTest(transform = ToTensor())
test_loader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = True)

print(cnn.accuracy(model=rgb_model, data_loader=test_loader))

cmat = conf_mat(test_loader, rgb_model)
#display_conf_mat(cmat, test_dataset.klass, 'rgb')

miss, miss_img = top_miss_classified(rgb_model, test_loader)
#plot_miss(miss, miss_img, 'rgb')


#RGB AUG
print(cnn.accuracy(model=rgb_aug_model, data_loader=test_loader))

cmat = conf_mat(test_loader, rgb_aug_model)
display_conf_mat(cmat, test_dataset.klass, 'rgb_aug')

miss, miss_img = top_miss_classified(rgb_aug_model, test_loader)
plot_miss(miss, miss_img, 'rgb_aug')


#GRAY
composed = torchvision.transforms.Compose([ToTensor(), ToGray()])

test_dataset = SignsTest(transform = composed)
test_loader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = True)

print(cnn.accuracy(model=gray_model, data_loader=test_loader))

cmat = conf_mat(test_loader, gray_model)
#display_conf_mat(cmat, test_dataset.klass, 'gray')

miss, miss_img = top_miss_classified(gray_model, test_loader)
#plot_miss(miss, miss_img, 'gray')


#GRAY AUG
print(cnn.accuracy(model=gray_aug_model, data_loader=test_loader))

cmat = conf_mat(test_loader, gray_aug_model)
display_conf_mat(cmat, test_dataset.klass, 'gray_aug')

miss, miss_img = top_miss_classified(gray_aug_model, test_loader)
plot_miss(miss, miss_img, 'gray_aug')




