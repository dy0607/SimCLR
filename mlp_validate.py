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

class Linear_Model(nn.Module):

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
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = ResNetSimCLR(**config["model"]).to(device)

# x = torch.randn(2, 3, 32, 32).to(device)
# print(model(x))
# exit()

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

lr = 0.001
num_epoch = 200
batch_size = 512
num_classes = 10

transform = transforms.Compose(
	[transforms.ToTensor(),]) # may need to rewrite

trainset = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True, transform = transform)
testset = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 3)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 3)

linear = Linear_Model(512, num_classes)
linear.to(device)

test_error = np.zeros(num_epoch)

for epoch in range(num_epoch):

	optimizer = optim.Adam(linear.parameters(), lr = lr * (0.1 ** (epoch // 100)), weight_decay = 1e-6)

	running_loss = train(linear, trainloader, optimizer, device, model)
	print('epoch %d: running loss = %.3f' % (epoch + 1, running_loss))

	with torch.no_grad():

		test_error[epoch], test_loss = test(linear, testloader, device, model)
		print('Test loss = %.3f, test error = %.3f %%\n' % (test_loss, test_error[epoch]))

print("best test error: %d" % (test_error.max()))
print(test_error)
