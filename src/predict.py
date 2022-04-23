from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, get_label_encoder
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

    model = Model(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4)

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
    trainer = Trainer(
        gpus=1,
        max_epochs=model.hparams.max_epochs,
    )
    consumer_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference=False),
                batch_size=32,
                shuffle=False,
            ),
        ],
    )
    consumer_output = torch.vstack([batch['image'] for batch in consumer_batch_output])
    #refs https://stackoverflow.com/a/952952
    consumer_output_imgs = [item for batch in consumer_batch_output for item in batch['image_path']]
    consumer_output_lbs = torch.cat([batch['label_id'] for batch in consumer_batch_output])
    consumer_output_lbs_raw = [item for batch in consumer_batch_output for item in batch['label']]

    reference_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference=True),
                batch_size=32,
                shuffle=False,
            ),
        ],
    )
    reference_output = torch.vstack([batch['image'] for batch in reference_batch_output])
    reference_output_imgs = [item for batch in reference_batch_output for item in batch['image_path']]
    reference_output_lbs = torch.cat([batch['label_id'] for batch in reference_batch_output])
    reference_output_real_labels = label_encoder.inverse_transform(reference_output_lbs.numpy())

    pos_mask = torch.zeros((consumer_output.shape[0], reference_output.shape[0]), dtype=torch.bool)
    for idx, label in enumerate(consumer_output_lbs):
        pos_mask[idx, label] = True

    # Similarity comparison
    cos_sim = F.cosine_similarity(consumer_output[:, None], reference_output[: None], dim=-1)

    # sorted_sim = torch.argsort(cos_sim, dim=1)

    label_columns = list(range(len(label_encoder.classes_)))
    df = pd.DataFrame(columns=['img_path', 'correct_label'] + label_columns)
    df['img_path'] = consumer_output_imgs
    df['correct_label'] = consumer_output_lbs
    df[label_columns] = cos_sim.cpu().numpy()
    df.to_csv('predicted.csv', index=False)