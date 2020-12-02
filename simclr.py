import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
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
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.rotation_criterion = nn.CrossEntropyLoss()

    def _get_device(self):
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, x, y, xis, xjs, n_iter):

        # r, z = model(x)
        # z = model.rotation_classifier(r)

        # # get the representations and the projections
        # #ris, zis = model(xis)  # [N,C]

        # # get the representations and the projections
        # #rjs, zjs = model(xjs)  # [N,C]

        # # normalize projection feature vectors
        # #zis = F.normalize(zis, dim=1)
        # #zjs = F.normalize(zjs, dim=1)
        
        # #loss = self.nt_xent_criterion(zis, zjs)
        # rotation_loss = self.rotation_criterion(z, y)        

        # #return loss + rotation_loss, loss
        # return rotation_loss, rotation_loss

        r0, _ = model(x)
        r0 = model.rotation_classifier(r0)
        
        z0 = torch.zeros(x.size(0)).long()
        z0 = z0.to(self.device)
        loss = self.rotation_criterion(r0, z0)
        print (r0.size(0), z0.size(0))

        r1, _ = model(y)
        r1 = model.rotation_classifier(r1)

        z1 = z0 + 1
        loss += self.rotation_criterion(r1, z1)

        r2, _ = model(xis)
        r2 = model.rotation_classifier(r2)

        z2 = z1 + 1
        loss += self.rotation_criterion(r2, z2)

        r3, _ = model(xjs)
        r3 = model.rotation_classifier(r3)

        z3 = z2 + 1
        loss += self.rotation_criterion(r3, z3)

        # print (x[0].sum(), y[0].sum(), r0[0], r1[0])

        return loss, loss


    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        # mt_train_loader, mt_valid_loader = self.mt_dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        # optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
        #                                                        last_epoch=-1)

        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for (x, y, xis, xjs), _ in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                    
                loss, _ = self._step(model, x, y, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

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
            # if epoch_counter >= 10:
            #     scheduler.step()
            #self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            contrastive_loss = 0.0
            rotation_loss = 0.0
            counter = 0

            correct = 0
            total = 0

            for (x, y, xis, xjs), _ in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss, _ = self._step(model, x, y, xis, xjs, counter)
                valid_loss += loss.item()

                contrastive_loss +=  _.item()
                rotation_loss += loss.item() - _.item()
                
                # validation accuracy
                # z, _ = model(x)
                # z = model.rotation_classifier(z)
                # predicted = torch.max(z, 1)[1]
                # total += y.size(0)
                # correct += (predicted == 0).sum().item()

                counter += 1

                r0, _ = model(x)
                r0 = model.rotation_classifier(r0)
                
                z = torch.zeros(x.size(0)).long()
                z = z.to(self.device)
                correct += (torch.max(r0, 1)[1] == z).sum().item()

                r1, _ = model(y)
                r1 = model.rotation_classifier(r1)

                z += 1
                correct += (torch.max(r1, 1)[1] == z).sum().item()

                r2, _ = model(xis)
                r2 = model.rotation_classifier(r2)

                z += 1
                correct += (torch.max(r2, 1)[1] == z).sum().item()

                r3, _ = model(xjs)
                r3 = model.rotation_classifier(r3)

                z += 1
                correct += (torch.max(r3, 1)[1] == z).sum().item()

                total += 4 * y.size(0)

            valid_loss /= counter
        model.train()

        print (valid_loss, contrastive_loss, rotation_loss, 100 * correct / total)

        return valid_loss
