from simclr import SimCLR
import yaml
from models.resnet_simclr import ResNetSimCLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# one layer linear model
class Linear_Model(nn.Module):

	def __init__(self, input_size, output_size):
		super(Linear_Model, self).__init__()
		self.fc = nn.Linear(input_size, output_size)

	def forward(self, x):
		x = self.fc(x)
		return x

# 2 layers MLP
class MLP(nn.Module):

	def __init__(self, input_size, output_size):
		super(Linear_Model, self).__init__()
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x

criterion = nn.CrossEntropyLoss() # cross entropy loss

def train(net, dataloader, optimizer, device, encoder):
	
	running_loss = 0
	tot = 0

	for i, data in enumerate(dataloader):
		images, labels = data[0].to(device), data[1].to(device)
		optimizer.zero_grad()
		
		with torch.no_grad():
			images = encoder(images)[0]

		out = net(images)
		loss = criterion(out, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		tot += 1

	return running_loss / tot

def test(net, dataloader, device, encoder):

	test_loss = 0
	tot = 0
	correct = 0

	for i, data in enumerate(dataloader):
		images, labels = data[0].to(device), data[1].to(device)
		images = encoder(images)[0]
		
		out = net(images)

		test_loss = (test_loss * i + criterion(out, labels).item()) / (i + 1)
		res = torch.argmax(out, 1)
		correct += (res == labels).sum().item()
		tot += labels.size(0)

	return 100 - 100 * correct / tot, test_loss


config_path = "./best_run/config.yaml"
model_path = "./best_run/model.pth"
data_path = "./data"

config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = ResNetSimCLR(**config["model"]).to(device)
model.load_state_dict(torch.load(model_path, map_location={'cuda:0':device, 'cuda:1':device, 'cuda:2':device, 'cuda:3':device}))

lr = 0.001
num_epoch = 90
batch_size = 256
num_classes = 10
weight_decay = 1e-6

training_set_size = 50000

base_transform = transforms.Compose(
	[transforms.ToTensor(), ])

train_transform = transforms.Compose(
	[transforms.RandomHorizontalFlip(),
	 transforms.RandomGrayscale(p=0.2),
	 transforms.ToTensor()])

linear = Linear_Model(512, num_classes) # Linear_Model or MLP
linear.to(device)

optimizer = optim.Adam(linear.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

trainset = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True, transform = train_transform)
trainset, valset = torch.utils.data.random_split(trainset, [training_set_size, 50000 - training_set_size])

testset = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True, transform = base_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 3)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 3)

test_error = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)

for epoch in range(num_epoch):

	running_loss = train(linear, trainloader, optimizer, device, model)
	print('epoch %d: running loss = %.3f' % (epoch + 1, running_loss))

	with torch.no_grad():

		test_error[epoch], test_loss = test(linear, testloader, device, model)
		print('Test loss = %.3f, test error = %.3f %%' % (test_loss, test_error[epoch]))

		train_error[epoch], train_loss = test(linear, trainloader, device, model)
		print('Training loss = %.3f, test error = %.3f %%\n' % (train_loss, train_error[epoch]))

	scheduler.step()

print("Best test error = %.3lf\n" % test_error.min())
print(test_error)
