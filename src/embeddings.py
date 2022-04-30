import argparse
from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, get_label_encoder, load_label_encoder
from torch.utils.data import DataLoader
from model import Model
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
from PIL import Image
from os import path
import os

from transforms import ToTensorD

if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataloader-num-workers', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model = Model.load_from_checkpoint(args.checkpoint)

    label_encoder = load_label_encoder(os.getenv('EPILLID_LABEL_ENCODER'))
    trainer = Trainer(
        gpus=args.gpus,
    )
    consumer_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference=False),
                batch_size=args.batch_size,
                num_workers=args.dataloader_num_workers,
                shuffle=False,
            ),
        ],
    )
    consumer_output = torch.vstack([batch['image'] for batch in consumer_batch_output])
    #refs https://stackoverflow.com/a/952952
    consumer_output_imgs = [item for batch in consumer_batch_output for item in batch['image_path']]

    reference_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference=True),
                batch_size=args.batch_size,
                num_workers=args.dataloader_num_workers,
                shuffle=False,
            ),
        ],
    )
    reference_output = torch.vstack([batch['image'] for batch in reference_batch_output])
    reference_output_imgs = [item for batch in reference_batch_output for item in batch['image_path']]

    label_columns = list(range(128))
    df = pd.DataFrame(columns=['img_path'] + label_columns)
    df['img_path'] = consumer_output_imgs + reference_output_imgs
    df[label_columns] = np.concatenate([consumer_output.cpu().numpy(), reference_output.cpu().numpy()])
    df.to_csv(f'{args.output}.csv', index=False)