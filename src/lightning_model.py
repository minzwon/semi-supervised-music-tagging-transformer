import math
import random
import torch
import torchaudio
import logging
import numpy as np
import pytorch_lightning as pl

from torch import nn
from sklearn import metrics

from msd_config import MSDConfig
from data_loader import msd_dataloader


class PLModel(pl.LightningModule):
    def __init__(
        self,
        data_path,
        network,
        teacher_network,
        loss_function,
        learning_rate,
        optimizer_class,
        batch_size,
        num_samples,
        num_chunks=1,
        num_workers=1,
        is_augmentation=False,
        is_expansion=False,
    ):
        """A customized PL model"""
        super().__init__()
        self.data_path = data_path
        self.network = network
        self.teacher_network = teacher_network
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.num_workers = num_workers
        self.is_augmentation = is_augmentation
        self.is_expansion = is_expansion

        self.save_hyperparameters()
        logging.info('Building pytorch lightning model - done')

        self.eval_logits = []
        self.eval_targets = []

    def train_dataloader(self):
        if self.is_expansion:
            labeled_loader = self.get_labeled_dataloader(data_split='TRAIN', 
                                                         batch_size=self.batch_size//2, 
                                                         is_agumentation=self.is_augmentation)
            unlabeled_loader = self.get_unlabeled_dataloader(batch_size=self.batch_size//2)
            loaders = {
                        'labeled': labeled_loader, 
                        'unlabeled': unlabeled_loader
                       }
            return loaders
        else:
            return self.get_labeled_dataloader(data_split='TRAIN', 
                                               batch_size=self.batch_size, 
                                               is_augmentation=self.is_augmentation)

    def val_dataloader(self):
        return self.get_labeled_dataloader(data_split='VALID', 
                                           batch_size=self.batch_size//self.num_chunks, 
                                           is_augmentation=False, 
                                           num_chunks=self.num_chunks)

    def test_dataloader(self):
        return self.get_labeled_dataloader(data_split='TEST',
                                           batch_size=self.batch_size//self.num_chunks,
                                           is_augmentation=False,
                                           num_chunks=self.num_chunks)

    def get_labeled_dataloader(self, data_split, batch_size, is_augmentation, num_chunks=1):
        return msd_dataloader(data_path=self.data_path, data_split=data_split, batch_size=batch_size, num_samples=self.num_samples, num_workers=self.num_workers, num_chunks=1, is_augmentation=is_augmentation)

    def get_unlabeled_dataloader(self, batch_size):
        return msd_dataloader(data_path=self.data_path, data_split='STUDENT', batch_size=batch_size, num_samples=self.num_samples, num_workers=self.num_workers, num_chunks=1)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        if self.is_expansion:
            wav, target_tag_binary = batch['labeled']
            wav = wav.squeeze(1)
            original_wav, noised_wav = batch['unlabeled']
            original_wav = original_wav.squeeze(1)
            noised_wav = noised_wav.squeeze(1)

            # concatenate audio for teacher / student
            teacher_input = torch.cat([wav, original_wav])
            student_input = torch.cat([wav, noised_wav])

            # get teacher prediction
            self.teacher_network.eval()
            with torch.no_grad():
                teacher_output = self.teacher_network.forward(teacher_input)
                teacher_prd = teacher_output[:len(wav)]
                pseudo_label = teacher_output[len(wav):]

            # forawrd student
            logits = self.network.forward(student_input)

            # loss functions
            teacher_loss = self.skeptical_loss(logits[:len(wav)], target_tag_binary, teacher_prd)
            student_loss = self.loss_function(logits[len(wav):], pseudo_label)
            loss = teacher_loss + student_loss

        else:
            wav, target_tag_binary = batch
            wav = wav.squeeze(1)
            logits = self.network.forward(wav)
            loss = self.loss_function(logits, target_tag_binary)

        self.log('train_loss_step', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        wav, target_tag_binary = batch
        b, c, t = wav.size()
        logits = self.network.forward(wav.view(-1, t))
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, target_tag_binary)
        self.log('valid_loss_step', loss, sync_dist=True)
        self.eval_logits.append(logits.detach().cpu())
        self.eval_targets.append(target_tag_binary.detach().cpu())

    def test_step(self, batch, batch_idx):
        wav, target_tag_binary = batch
        b, c, t = wav.size()
        logits = self.network.forward(wav.view(-1, t))
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, target_tag_binary)
        self.log('test_loss_step', loss, on_epoch=True)
        self.eval_logits.append(logits.detach().cpu())
        self.eval_targets.append(target_tag_binary.detach().cpu())

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.network.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_epoch_end(self, outputs):
        logits = torch.cat(self.eval_logits, dim=0)
        target_binaries = torch.cat(self.eval_targets, dim=0)
        loss = self.loss_function(logits, target_binaries)
        roc_auc, pr_auc = self.get_auc_scores(logits, target_binaries)
        self.log('valid_loss', loss.cuda(), sync_dist=True)
        self.log('valid_roc_auc', roc_auc.cuda(), sync_dist=True)
        self.log('valid_pr_auc', pr_auc.cuda(), sync_dist=True)
        self.eval_logits = []
        self.eval_targets = []

    def test_epoch_end(self, outputs):
        logits = torch.cat(self.eval_logits, dim=0)
        target_binaries = torch.cat(self.eval_targets, dim=0)
        loss = self.loss_function(logits, target_binaries)
        roc_auc, pr_auc = self.get_auc_scores(logits, target_binaries)
        self.eval_logits = []
        self.eval_targets = []

    def get_auc_scores(self, logits, targets):
        try:
            roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
            pr_auc = metrics.average_precision_score(targets, logits, average='macro')
            print('roc_auc: %.4f' % roc_auc)
            print('pr_auc: %.4f' % pr_auc)

            # tag wise score
            roc_aucs = metrics.roc_auc_score(targets, logits, average=None)
            pr_aucs = metrics.average_precision_score(targets, logits, average=None)
            for i in range(50):
                print('%s: %.4f, %.4f' % (MSDConfig.tag_names[i], roc_aucs[i], pr_aucs[i]))
        except ValueError as e:
            roc_auc, pr_auc = 0, 0
            print('auc not available')
        return torch.tensor(roc_auc), torch.tensor(pr_auc)
