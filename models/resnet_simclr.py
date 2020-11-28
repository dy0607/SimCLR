import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

	def __init__(self, base_model, out_dim):
		super(ResNetSimCLR, self).__init__()
		self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
							"resnet50": models.resnet50(pretrained=False)}
							
		resnet = self._get_basemodel(base_model)

		# change the original resnet18 to accomondate CIFAR10
		resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, ceil_mode=False)

		num_ftrs = resnet.fc.in_features

		self.features = nn.Sequential(*list(resnet.children())[:-1])

		# projection MLP
		self.l1 = nn.Linear(num_ftrs, num_ftrs)
		self.l2 = nn.Linear(num_ftrs, out_dim)

	def _get_basemodel(self, model_name):
		try:
			model = self.resnet_dict[model_name]
			print("Feature extractor:", model_name)
			return model
		except:
			raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

	def forward(self, x):
		h = self.features(x)
		h = h.squeeze()

		x = self.l1(h)
		x = F.relu(x)
		x = self.l2(x)
		return h, x
