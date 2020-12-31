import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets

np.random.seed(0)

class DataSetWrapper(object):

	def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.valid_size = valid_size
		self.s = s
		self.input_shape = eval(input_shape)

	def get_data_loaders(self):

		data_augment, base_transform = self._get_simclr_pipeline_transforms()
		
		# train dataset, for contrastive learning
		train_dataset = datasets.CIFAR10('./data', train=True, download=True,
									   transform=SimCLRDataTransform(data_augment, base_transform))

		# the original dataset, for clustering
		original_dataset = datasets.CIFAR10('./data', train=True, download=True,
									   transform=base_transform)

		train_sampler, valid_sampler = self.get_train_validation_samplers(len(train_dataset))

		clustering_loader = self._get_data_loader(original_dataset, train_sampler)
		train_loader = self._get_data_loader(train_dataset, train_sampler)
		valid_loader = self._get_data_loader(train_dataset, valid_sampler)
		
		return train_loader, valid_loader, clustering_loader
	
	def _get_data_loader(self, dataset, sampler):

		dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
								num_workers=self.num_workers, drop_last=True, shuffle=False)

		return dataloader


	def _get_simclr_pipeline_transforms(self):
		# get a set of data augmentation transformations as described in the SimCLR paper.
		color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)

		# for CIFAR10, leave out the gaussian blur
		data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
											  transforms.RandomHorizontalFlip(),
											  transforms.RandomApply([color_jitter], p=0.8),
											  transforms.RandomGrayscale(p=0.2),
											  # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
											  transforms.ToTensor()])

		# base transform, used for clustering
		base_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
										    transforms.RandomApply([color_jitter], p=0.5),
										  # transforms.RandomGrayscale(p=0.2),
										  # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
										  transforms.ToTensor()])


		return data_transforms, base_transform

	def get_train_validation_samplers(self, num_train):
		# obtain training indices that will be used for validation
		indices = list(range(num_train))
		np.random.shuffle(indices)

		split = int(np.floor(self.valid_size * num_train))
		train_idx, valid_idx = indices[split:], indices[:split]

		# define samplers for obtaining training and validation batches
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		return train_sampler, valid_sampler

class SimCLRDataTransform(object):
	def __init__(self, transform, base):
		self.transform = transform
		self.base = base

	def __call__(self, sample):

		# positive pair
		xi = self.transform(sample)
		xj = self.transform(sample)

		# base image
		x = self.base(sample)

		return x, xi, xj
