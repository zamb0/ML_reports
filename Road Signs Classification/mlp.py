import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import os
import matplotlib.pyplot as plt

import inspect_data as insp

device = torch.device('mps')


class LowFeaturesTrain(Dataset):
	def __init__(self, transform = None, name = None):
		x, _, y, _, klass = insp.import_features(name)
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
	
class LowFeaturesTest(Dataset):
	def __init__(self, transform = None, name = None):
		_, x, _, y, klass  = insp.import_features(name)
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
				return inputs, labels
		
class MLPNet(nn.Module):
	def __init__(self, input_size, num_classes):
		super(MLPNet,self).__init__()
		self.input_size = input_size
		self.num_classes = num_classes

		self.fc1 = nn.Linear(self.input_size, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, self.num_classes)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		
		return x
	
	def shape(self):
		return np.array([self.input_size, self.input_size*4, self.input_size*2, self.input_size//2, self.num_classes])
	

def accuracy(model, data_loader):
		""" calculate accuracy

		Parameters
		----------
			model : torch.nn.Module
				model to be tested
			data_loader : DataLoader
				data loader

		Returns
		-------
			acc : float
				accuracy
		"""
		
		n_correct = 0
		n_samples = 0
		for features, labels in data_loader:
				features = features.to(device)
				labels = labels.to(device)
				outputs = model(features)
				# max returns (value ,index)
				_, predicted = torch.max(outputs.data, 1)
				n_samples += labels.size(0)
				n_correct += (predicted == labels).sum().item()

		acc = 100.0 * n_correct / n_samples
		#print(f'Accuracy: {acc} %')

		return acc


if __name__ == "__main__":
		
		#ft = import_features()
		#print(ft[0].shape)
		#print(ft[1].shape)
		#print(ft[2].shape)

		f_methods = [f[:-13] for f in os.listdir('low_features') if f.endswith('_train.txt.gz')]
		f_methods.sort()
		#print(f_methods)

		train_dataset = LowFeaturesTrain(transform = ToTensor(), name = f_methods[2])
		test_dataset = LowFeaturesTest(transform = ToTensor(), name = f_methods[2])

		input_size = train_dataset.x.shape[1]
		num_classes = len(np.unique(train_dataset.y))
		num_epochs = 200
		learning_rate = 0.001
		batch_size = train_dataset.__len__()//num_epochs

		train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
		test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
		
		model = MLPNet(input_size, num_classes).to(device)

		print(model.shape())
					
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


		acc_train_list = np.array([])
		acc_test_list = np.array([])

		# Train the model
		n_total_steps = len(train_loader)
		for epoch in range(num_epochs):
				for i, (features, labels) in enumerate(train_loader):  
				
						# Forward pass
						features = features.to(device) 
						outputs = model(features)
						loss = criterion(outputs, labels)
						
						# Backward and optimize
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()

				
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
						
				print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {acc_train:.4f}%, Test Accuracy: {acc_test:.4f}%')
		

		PATH = './mlp.pth'
		torch.save(model.state_dict(), PATH)

		plt.plot(acc_train_list)
		plt.plot(acc_test_list)
		plt.title("Accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Train vs Test")
		plt.legend(["Train", "Test"])
		plt.grid()
		plt.savefig("img/mlp.png", dpi=300)
		plt.show()