from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDSingleTypeDataset, get_label_encoder
from torch.utils.data import DataLoader
from model import Model
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import numpy as np
import torch
import os

from transforms import ToTensorD

if __name__ == '__main__':
    load_dotenv()

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
    # ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    # ds = EPillIDDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder=label_encoder, transforms=ToTensorD(keys=['consumer', 'reference']))
    # dl = DataLoader(ds, batch_size=32)
    model = Model(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4)
    trainer = Trainer(
        gpus=1,
        max_epochs=model.hparams.max_epochs,
    )
    # trainer.fit(model, train_dataloaders=[dl])
    consumer_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference_set=False, transforms=ToTensor()),
                batch_size=32,
                shuffle=False,
            ),
        ],
    )
    consumer_output = torch.vstack(consumer_batch_output)

    reference_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, use_reference_set=True, transforms=ToTensor()),
                batch_size=32,
                shuffle=False,
            ),
        ],
    )
    reference_output = torch.vstack(reference_batch_output)

    pos_mask = torch.zeros((consumer_output.shape[0], reference_output.shape[0]), dtype=torch.bool)
    for i in range(consumer_output.shape[0]):
        pos_mask[i, 0] = True #TODO get label_id

    # Similarity comparison
    cos_sim = F.cosine_similarity(consumer_output[:, None], reference_output[: None], dim=-1)
    # Move the positive samples to the front of the array for convenience
    comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # Correct reference image
        dim=-1,
    )

    #TODO add metadata to datasets
    #TODO metric calculation