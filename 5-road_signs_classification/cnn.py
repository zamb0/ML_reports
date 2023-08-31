from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import inspect_data as insp

device = torch.device('mps')

class SignsTrain(Dataset):
	def __init__(self, transform = None):
		x, y, _, _, klass = insp.make_dataset("road-signs")
		self.x = x
		self.y = y
		self.klass = klass
		self.transform = transform

	def __getitem__(self, index):
		#print("Index: ",index)
		sample = self.x[index],self.y[index]
		if self.transform:
			sample = self.transform(sample)
		return sample

	def __len__(self):
		return self.x.shape[0]



class SignsTest(Dataset):
	def __init__(self, transform = None):
		_, _, x, y, klass = insp.make_dataset("road-signs")
		self.x = x
		self.y = y
		self.klass = klass
		self.transform = transform

	def __getitem__(self, index):
		#print("Index: ",index)
		sample = self.x[index],self.y[index]
		if self.transform:
			sample = self.transform(sample)
		return sample

	def __len__(self):
		return self.x.shape[0]



class ToTensor:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = np.array(inputs, dtype = np.float32)
		labels = np.array(labels, dtype = np.int64)
		inputs = torch.from_numpy(inputs)
		labels = torch.from_numpy(labels)
		inputs = inputs.permute(2,0,1)
		return inputs, labels
  
class ToGray:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = torchvision.transforms.functional.rgb_to_grayscale(inputs)
		return inputs, labels
	
class ToEraseBlock:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)(inputs)

		return inputs, labels

class ToGaussianBlur:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(inputs)

		return inputs, labels
	
class ToGaussianNoise:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = noise(inputs)

		return inputs, labels

class ColorConvNet(nn.Module):
	def __init__(self):
		super(ColorConvNet,self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*47*47, 128)
		self.fc2 = nn.Linear(128, 84)
		self.fc3 = nn.Linear(84, 20)


	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 47 * 47)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))    
		x = self.fc3(x)

		return x

	def forward_shape(self,x):
		shape = np.array([])
		print("Input shape: ", x.shape)
		x = self.conv1(x)
		print("Conv1 shape: ", x.shape)
		x = self.pool(x)
		print("Pool1 shape: ", x.shape)
		x = F.relu(x)
		print("Relu1 shape: ", x.shape)
		x = self.conv2(x)
		print("Conv2 shape: ", x.shape)
		x = self.pool(x)
		print("Pool2 shape: ", x.shape)
		x = F.relu(x)
		print("Relu2 shape: ", x.shape)
		x = x.view(-1, 16 * 47 * 47)
		print("View shape: ", x.shape)
		x = self.fc1(x)
		print("FC1 shape: ", x.shape)
		x = F.relu(x)
		print("Relu3 shape: ", x.shape)
		x = self.fc2(x)
		print("FC2 shape: ", x.shape)
		x = F.relu(x)
		print("Relu4 shape: ", x.shape)
		x = self.fc3(x)
		print("FC3 shape: ", x.shape)

  
class GrayConvNet(nn.Module):
	def __init__(self):
		super(GrayConvNet,self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*47*47, 128)
		self.fc2 = nn.Linear(128, 84)
		self.fc3 = nn.Linear(84, 20)


	def forward(self,x):

		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 47 * 47)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))    
		x = self.fc3(x)

		return x
	
	def forward_shape(self,x):
		shape = np.array([])
		print("Input shape: ", x.shape)
		x = self.conv1(x)
		print("Conv1 shape: ", x.shape)
		x = self.pool(x)
		print("Pool1 shape: ", x.shape)
		x = F.relu(x)
		print("Relu1 shape: ", x.shape)
		x = self.conv2(x)
		print("Conv2 shape: ", x.shape)
		x = self.pool(x)
		print("Pool2 shape: ", x.shape)
		x = F.relu(x)
		print("Relu2 shape: ", x.shape)
		x = x.view(-1, 16 * 47 * 47)
		print("View shape: ", x.shape)
		x = self.fc1(x)
		print("FC1 shape: ", x.shape)
		x = F.relu(x)
		print("Relu3 shape: ", x.shape)
		x = self.fc2(x)
		print("FC2 shape: ", x.shape)
		x = F.relu(x)
		print("Relu4 shape: ", x.shape)
		x = self.fc3(x)
		print("FC3 shape: ", x.shape)
  

def noise(img):
	""" Gaussian Noise 

	Parameters
	----------
	img : torch.Tensor
		Input image	

	Returns
	-------
	img : torch.Tensor
		Output image

	"""

	noise = torch.randn(img.size()) * (0.1**0.5)

	img = img + noise
	img = torch.clamp(img, 0, 1)
	return img


def accuracy(model, data_loader):
	""" calculate accuracy	

	Parameters
	----------
		model : torch.nn.Module
			model to be tested
		data_loader : DataLoader
			test data loader

	Returns
	-------
		acc : float
			accuracy
	"""

	with torch.no_grad():
		n_correct = 0
		n_samples = 0

		for images, labels in data_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)

			_, predicted = torch.max(outputs.data, 1)
			n_samples += labels.size(0)
			n_correct += (predicted==labels).sum().item()
  
		acc = n_correct / n_samples

	return np.array(acc)


def train_model(train_dataset, test_dataset, model, params = [10, 1e-2]):
	""" train model

	Parameters
	----------
		train_dataset : Dataset
			train dataset
		test_dataset : Dataset
			test dataset
		model : torch.nn.Module
			model to be trained
		params : list
			hyperparameters

	Returns
	-------
		model : torch.nn.Module
			trained model
		acc_train_list : np.array
			train accuracy list
		acc_test_list : np.array
			test accuracy list
	"""

	num_epochs = params[0]
	learning_rate = params[1]
	batch_size = 32

	train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle =True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
	

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), learning_rate)

	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) 

	acc_train_list = np.array([])
	acc_test_list = np.array([])

	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):

			#for i, img in enumerate(images):
			#	plt.subplot(4,8,i+1)
			#	plt.imshow(img.permute(1,2,0))
			#	plt.axis('off')
			#plt.show()

			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)

			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()         
			  
		#lr_scheduler.step() 
		with torch.no_grad():
			acc_train = accuracy(model, train_loader)

			if acc_train_list.size != 0:
				acc_train_list = np.concatenate((acc_train_list, np.array([acc_train])), axis = 0)
			else:
				acc_train_list = np.array([acc_train])

			acc_test = accuracy(model, test_loader)

			if acc_test_list.size != 0:
				acc_test_list = np.concatenate((acc_test_list, np.array([acc_test])), axis = 0)
			else:
				acc_test_list = np.array([acc_test])
		
		print(f"Epoch [{epoch+1}/{num_epochs}]: loss = {loss:.5f}, train_acc = {acc_train*100:.5f}, test_acc = {acc_test*100:.5f}")

	return model, acc_train_list, acc_test_list


def train_Color_model(augmentation = False, params = [20, 1e-3]):
	""" train color model

	Parameters
	----------
		augmentation : bool
			whether to use augmentation or not
		params : list
			hyperparameters

	Returns
	-------
		acc_train : np.array
			train accuracy list
		acc_test : np.array
			test accuracy list
	"""
	model = ColorConvNet().to(device)

	if augmentation:
		composed_aug = torchvision.transforms.Compose([
							ToTensor(), 
							ToEraseBlock()])
		
		composed_aug2 = torchvision.transforms.Compose([
							ToTensor(), 
							ToGaussianNoise()])
		
		composed = torchvision.transforms.Compose([
							ToTensor()])
		
		train_dataset = SignsTrain(transform = composed)
		train_dataset = torch.utils.data.ConcatDataset([SignsTrain(transform = composed), SignsTrain(transform = composed_aug), SignsTrain(transform = composed_aug2)])

		test_dataset = SignsTest(transform = ToTensor())
	else:
		train_dataset = SignsTrain(transform = ToTensor())
		test_dataset = SignsTest(transform = ToTensor())


	model, acc_train, acc_test = train_model(train_dataset, test_dataset, model, params = params)

	if augmentation:
		PATH = './rgb_aug_cnn.pth'
	else:
		PATH = './rgb_cnn.pth'
	#torch.save(model.state_dict(), PATH)

	return acc_train, acc_test



def train_Gray_model(augmentation = False, params = [20, 1e-3]):
	""" train gray model

	Parameters
	----------
		augmentation : bool
			whether to use augmentation or not
		params : list
			hyperparameters

	Returns
	-------
		acc_train : np.array
			train accuracy list
		acc_test : np.array
			test accuracy list
	"""
	model = GrayConvNet().to(device)

	if augmentation:
		composed_aug = torchvision.transforms.Compose([
						ToTensor(), 
					    ToGray(), 
						ToEraseBlock()])
		
		composed_aug2 = torchvision.transforms.Compose([
						ToTensor(), 
					    ToGray(), 
						ToGaussianNoise()])
		
		composed = torchvision.transforms.Compose([
						ToTensor(), 
					    ToGray()])
		
		train_dataset = torch.utils.data.ConcatDataset([SignsTrain(transform = composed), SignsTrain(transform = composed_aug), SignsTrain(transform = composed_aug2)])

		composed = torchvision.transforms.Compose([ToTensor(), ToGray()])
		test_dataset = SignsTest(transform = composed)
	else:
		composed = torchvision.transforms.Compose([ToTensor(), ToGray()])
		train_dataset = SignsTrain(transform = composed)
		test_dataset = SignsTest(transform = composed)


	model, acc_train, acc_test = train_model(train_dataset, test_dataset, model, params = params)

	if augmentation:
		PATH = './gray_aug_cnn.pth'
	else:
		PATH = './gray_cnn.pth'
	#torch.save(model.state_dict(), PATH)

	return acc_train, acc_test


def ex_aug(img):
	""" example of augmentation

	Parameters
	----------
		img : torch.Tensor
			input image
	"""

	plt.figure(figsize=(10, 4))
	plt.subplot(1, 3, 1)
	plt.imshow(img.permute(1,2,0))
	plt.axis('off')
	plt.title("Original")

	plt.subplot(1, 3, 2)
	plt.imshow(torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)(img).permute(1,2,0))
	plt.axis('off')
	plt.title("Random Erasing")

	plt.subplot(1, 3, 3)
	plt.imshow(noise(img).permute(1,2,0))
	plt.axis('off')
	plt.title("Gaussian Blur")

	plt.tight_layout()
	plt.savefig("img/aug.png", dpi=300)
	plt.show()





if __name__ == "__main__":

	#ColorConvNet().forward_shape(torch.rand(1,3,200,200))
	#GrayConvNet().forward_shape(torch.rand(1,1,200,200))
	
	ex_aug(torchvision.transforms.ToTensor()(plt.imread("road-signs/prohibitory-no-overtaking/train/11.jpg")))

	acc_train_list, acc_test_list = train_Color_model(augmentation = True, params=[20, 1e-3])

	plt.plot(acc_train_list*100)
	plt.plot(acc_test_list*100)
	plt.title("Accuracy")
	plt.ylim(50, 100)
	plt.xlabel("Epoch")
	plt.ylabel("Train vs Test")
	plt.xticks(np.arange(0, 20, 2))
	plt.legend(["Train", "Test"])
	plt.grid()
	plt.savefig("img/color_aug_cnn.png", dpi=300)
	plt.show()

	acc_train_list, acc_test_list = train_Gray_model(augmentation = True, params=[20, 1e-3])

	plt.plot(acc_train_list*100)
	plt.plot(acc_test_list*100)
	plt.title("Accuracy")
	plt.ylim(50, 100)
	plt.xlabel("Epoch")
	plt.ylabel("Train vs Test")
	plt.xticks(np.arange(0, 20, 2))
	plt.legend(["Train", "Test"])
	plt.grid()
	plt.savefig("img/gray_aug_cnn.png", dpi=300)
	plt.show()




