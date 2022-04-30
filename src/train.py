from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, EPillIDSupervisedContrastiveDataset, get_label_encoder, load_label_encoder
from torch.utils.data import DataLoader
from model import Model
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import os
import argparse

from transforms import ToTensorD

# def collate_fn(batch):
#     dict = {}
#     for item in batch:
#         for key, value in item.items():
#             if not key in dict:
#                 dict[key] = [value]
#             else:
#                 dict[key].append(value)
#     for key, values in dict.items():
#         if torch.is_tensor(values[0]):
#             dict[key] = torch.stack(values)
#         elif type(values[0]).__name__ == 'ndarray':
#             dict[key] = torch.stack([torch.tensor(element) for element in values])
#     return dict

if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataloader-num-workers', type=int, default=1)
    parser.add_argument('--pretraining', default=False, action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--pretrain-from-checkpoint')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--checkpoint-name')
    args = parser.parse_args()

    load_model_checkpoint = None
    if args.pretrain_from_checkpoint != None:
        load_model_checkpoint = args.pretrain_from_checkpoint
    elif args.resume_from_checkpoint != None:
        load_model_checkpoint = args.resume_from_checkpoint

    # Load the model early in order to grab hyperparameters. This is probably not the most efficient
    # operation, but it is convenient
    model = Model.load_from_checkpoint(load_model_checkpoint) \
        if load_model_checkpoint != None else \
        Model(
            hidden_dim=128, 
            lr=5e-4, 
            temperature=0.07,
            weight_decay=1e-4, 
            pretraining=args.pretraining, 
            fold=args.fold, 
            batch_size=args.batch_size, 
            max_epochs=args.epochs,
            checkpoint_path=args.checkpoint_path,
            checkpoint_name=args.checkpoint_name,
        )
    
    if args.pretrain_from_checkpoint != None:
        # fill in custom hparams
        model.hparams['hidden_dim']=128
        model.hparams['lr']=5e-4
        model.hparams['temperature']=0.07
        model.hparams['weight_decay']=1e-4
        model.hparams['pretraining']=args.pretraining
        model.hparams['fold']=args.fold
        model.hparams['batch_size']=args.batch_size
        model.hparams['max_epochs']=args.epochs
        model.hparams['checkpoint_path']=args.checkpoint_path
        model.hparams['checkpoint_name']=args.checkpoint_name
    
    label_encoder = load_label_encoder(os.getenv('EPILLID_LABEL_ENCODER'))

    dataset = \
        C3PIDTDMixSelfSupervisedContrastiveDataset(
            os.getenv('DTD_DATASET_ROOT'), 
            os.getenv('C3PI_REFERENCE_PILLS')
        ) \
        if model.hparams.pretraining else \
        EPillIDSupervisedContrastiveDataset(
            os.getenv('EPILLID_DATASET_ROOT'), 
            label_encoder, 
            fold=model.hparams.fold, 
            validation=False,
        )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=model.hparams.batch_size,
        num_workers=args.dataloader_num_workers,
    )
   
    checkpoint = ModelCheckpoint(
        monitor="training_loss", 
        mode="min", 
        save_top_k=1, 
        save_last=True,
        dirpath=model.hparams.checkpoint_path,
        filename=model.hparams.checkpoint_name,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f'{model.hparams.checkpoint_name}-logs/'
    )
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=model.hparams.max_epochs,
        callbacks=[checkpoint],
        limit_val_batches=1 if model.hparams.pretraining else 1.0,
        logger=tb_logger,
    )
    trainer.fit(
        model, 
        train_dataloaders=[dataloader],
        val_dataloaders=[dataloader],
        ckpt_path=args.resume_from_checkpoint,
    )