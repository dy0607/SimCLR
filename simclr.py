import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss

from sklearn.cluster import KMeans, MiniBatchKMeans

import os
import shutil
import sys

apex_support = False
try:
	sys.path.append('./apex')
	from apex import amp

	apex_support = True
except:
	print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
	apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
	if not os.path.exists(model_checkpoints_folder):
		os.makedirs(model_checkpoints_folder)
		shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

	def __init__(self, dataset, config):
		self.config = config
		self.device = self._get_device()
		self.writer = SummaryWriter()
		self.dataset = dataset
		
		self.criterion = torch.nn.KLDivLoss(reduction="batchmean")
		self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

	def _get_device(self):
		device = self.config['device'] if torch.cuda.is_available() else 'cpu'
		print("Running on:", device)
		return device

	def _step(self, model, x, xis, xjs, n_iter, train = True):

		# get the representations and the projections
		ris, zis = model(xis)
		rjs, zjs = model(xjs)
		rx, px = model(x)

		# normalize projection feature vectors
		zis = F.normalize(zis, dim=1)
		zjs = F.normalize(zjs, dim=1)
		px = F.normalize(px, dim=1)

		labels = F.softmax(torch.matmul(px, self.C), dim=1)
		dist, clusters = labels.max(dim=1)
		loss0 = -dist.mean()

		pi = F.log_softmax(torch.matmul(zis, self.C), dim=1)
		pj = F.log_softmax(torch.matmul(zjs, self.C), dim=1)
		# print(pi, pj, labels)
		
		loss1 = self.criterion(pi, labels) + self.criterion(pj, labels)
		loss2 = self.nt_xent_criterion(zis, zjs)

		# print("x", loss1)
		# print(loss2)
		# exit()

		if n_iter % self.config['log_every_n_steps'] == 0 and train:
			self.writer.add_scalar('loss0', -loss0, global_step=n_iter)
			self.writer.add_scalar('loss1', loss1, global_step=n_iter)

		return loss0 + loss1 + loss2

	def _get_clustering(self, model, dataloader):

		with torch.no_grad():

			print('Clustering...')

			kmeans = 0
			score = 0

			if not self.config['minibatch_kmeans']:
			
				features = []
				for x, _ in dataloader:
					x = x.to(self.device)
					_, projections = model(x)
					new_features = F.normalize(projections, dim=1).cpu().numpy()
					features.append(new_features)
					
				features = np.concatenate(features, axis=0)
				kmeans = KMeans(n_clusters=self.config['n_clusters']).fit(features)
				score = kmeans.score(features)

			else:

				kmeans = MiniBatchKMeans(n_clusters=self.config['n_clusters'], 
										batch_size=self.config['batch_size'])
				for x, _ in dataloader:
					x = x.to(self.device)
					_, projections = model(x)
					features = F.normalize(projections, dim=1).cpu().numpy()
					kmeans.partial_fit(features)
					score += kmeans.score(features)
			
			centers = torch.tensor(kmeans.cluster_centers_)
			centers = F.normalize(centers, dim=1).to(self.device)
			self.C = centers.transpose_(0, 1)
			
			count = [0 for i in range(self.config['n_clusters'])]

			for i in range(len(kmeans.labels_)):
				count[kmeans.labels_[i]] += 1

			self.writer.add_scalar('clustering score', score, global_step=self.cluster_n_iter)
			self.cluster_n_iter += 1

			print(count)
			print(self.C)

	def train(self):

		train_loader, valid_loader, clustering_loader = self.dataset.get_data_loaders()

		model = ResNetSimCLR(**self.config["model"]).to(self.device)
		model = self._load_pre_trained_weights(model)

		optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
															   last_epoch=-1)

		if apex_support and self.config['fp16_precision']:
			model, optimizer = amp.initialize(model, optimizer,
											  opt_level='O2',
											  keep_batchnorm_fp32=True)

		model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

		# save config file
		_save_config_file(model_checkpoints_folder)

		# reset the counters
		self.n_iter = 0
		valid_n_iter = 0

		self.cluster_n_iter = 0
		best_valid_loss = np.inf

		for epoch_counter in range(self.config['epochs']):

			if epoch_counter % self.config['cluster_every_n_epochs'] == 0:
				self._get_clustering(model, clustering_loader)

			self._train(model, optimizer, train_loader)

			# validate the model if requested
			if epoch_counter % self.config['eval_every_n_epochs'] == 0:
				valid_loss = self._validate(model, valid_loader)
				if valid_loss < best_valid_loss:
					# save the model weights
					best_valid_loss = valid_loss
					torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

				self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
				valid_n_iter += 1

			# warmup for the first 10 epochs
			if epoch_counter >= 10:
				scheduler.step()
			self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=self.n_iter)

	def _load_pre_trained_weights(self, model):
		try:
			checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
			state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
			model.load_state_dict(state_dict)
			print("Loaded pre-trained model with success.")
		except FileNotFoundError:
			print("Pre-trained weights not found. Training from scratch.")

		return model

	def _train(self, model, optimizer, train_loader):
		for (x, xis, xjs), _ in train_loader:
			
			optimizer.zero_grad()

			x = x.to(self.device)
			xis = xis.to(self.device)
			xjs = xjs.to(self.device)

			loss = self._step(model, x, xis, xjs, self.n_iter)

			if self.n_iter % self.config['log_every_n_steps'] == 0:
				self.writer.add_scalar('train_loss', loss, global_step=self.n_iter)

			if apex_support and self.config['fp16_precision']:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			self.n_iter += 1

	def _validate(self, model, valid_loader):

		# validation steps
		with torch.no_grad():
			model.eval()

			valid_loss = 0.0
			counter = 0
			for (x, xis, xjs), _ in valid_loader:
				
				x = x.to(self.device)
				xis = xis.to(self.device)
				xjs = xjs.to(self.device)

				loss = self._step(model, x, xis, xjs, counter, train = False)
				valid_loss += loss.item()
				counter += 1
			valid_loss /= counter
		model.train()
		return valid_loss
