import os
import torch
import logging
import argparse
import numpy as np
import pytorch_lightning as pl

from torch import nn
from models import MusicTaggingTransformer
from lightning_model import PLModel
from callbacks_loggers import get_loggers, get_callbacks
from training_utils import DirManager


def train(args):
    dir_manager = DirManager(output_dir=args.output_dir)

    # model
    _network = MusicTaggingTransformer(
        conv_ndim=args.conv_ndim,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        f_min=args.f_min,
        f_max=args.f_max,
        attention_ndim=args.attention_ndim,
        attention_nheads=args.attention_nheads,
        attention_nlayers=args.attention_nlayers,
        attention_max_len=args.attention_max_len,
        dropout=args.dropout,
        n_seq_cls=args.n_seq_cls,
        n_token_cls=args.n_token_cls,
    )
    if args.is_expansion:
        _teacher_network = MusicTaggingTransformer(
            conv_ndim=args.teacher_conv_ndim,
            n_mels=args.n_mels,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            f_min=args.f_min,
            f_max=args.f_max,
            attention_ndim=args.teacher_attention_ndim,
            attention_nheads=args.teacher_attention_nheads,
            attention_nlayers=args.teacher_attention_nlayers,
            attention_max_len=args.attention_max_len,
            dropout=args.dropout,
            n_seq_cls=args.n_seq_cls,
            n_token_cls=args.n_token_cls,
        )
        S = torch.load(args.teacher_model_path)
        S = {k[8:]: v for k, v in S.items() if k[:7]!='teacher'}
        _teacher_network.load_state_dict(S)
    else:
        _teacher_network = nn.Module()

    # count gpus
    num_gpus = torch.cuda.device_count()

    # data loader
    callbacks = get_callbacks(patience=40, dir_manager=dir_manager, monitor='valid_loss')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, save_top_k=1, verbose=True, monitor='valid_loss', mode='min') # prefix='' 
    loggers = get_loggers(tb_save_dir=dir_manager.tensorboard_dir)

    # loss function
    model = PLModel(
        data_path=args.data_path,
        network=_network,
        teacher_network=_teacher_network,
        loss_function=nn.BCELoss(),
        learning_rate=args.learning_rate,
        optimizer_class=torch.optim.Adam,
        batch_size=args.batch_size,
        num_samples=args.input_length,
        num_chunks=args.num_chunks,
        num_workers=args.num_workers,
        is_augmentation=args.is_augmentation,
        is_expansion=args.is_expansion,
    )

    # trainer
    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=args.num_nodes,
        logger=loggers,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint_callback,
        sync_batchnorm=True,
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=None,
        num_sanity_val_steps=2,
        automatic_optimization=True,
        replace_sampler_ddp=False,
        accelerator="ddp",
        multiple_trainloader_mode='max_size_cycle',
    )
    trainer.fit(model)
    logging.info('Training is done. Exporting the best model..')

    # load the best model and save it to onnx file
    model = PLModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    torch.save(model.state_dict(), dir_manager.best_model_statedict)
    logging.info('Best performing model was exported to onnx and torchscript.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='TRAIN', choices=['TRAIN', 'TEST'])
    parser.add_argument('--is_augmentation', type=bool, default=False)
    parser.add_argument('--is_expansion', type=bool, default=False)

    # stft parameters
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=11025)
    parser.add_argument('--sample_rate', type=int, default=22050)

    # input parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('--input_length', type=int, default=220500)

    # model parameters
    parser.add_argument('--conv_ndim', type=int, default=128)
    parser.add_argument('--attention_ndim', type=int, default=256)
    parser.add_argument('--attention_nheads', type=int, default=8)
    parser.add_argument('--attention_nlayers', type=int, default=4)
    parser.add_argument('--attention_max_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_seq_cls', type=int, default=50)
    parser.add_argument('--n_token_cls', type=int, default=1)

    # teacher model parameters
    parser.add_argument('--teacher_conv_ndim', type=int, default=128)
    parser.add_argument('--teacher_attention_ndim', type=int, default=256)
    parser.add_argument('--teacher_attention_nheads', type=int, default=8)
    parser.add_argument('--teacher_attention_nlayers', type=int, default=4)

    # training parameters
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # load / save paths
    parser.add_argument('--data_path', type=str, default='./../data/')
    parser.add_argument('--output_dir', type=str, default='./results/exp')
    parser.add_argument('--teacher_model_path', type=str, default='./results/exp')


    args = parser.parse_args()

    print(args)
    train(args)
