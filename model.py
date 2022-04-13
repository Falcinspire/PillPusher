#refs https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/13-contrastive-learning.html

from sched import scheduler
from torchvision.models import resnet18
import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule, Trainer

import matplotlib.pyplot as plt

class Model(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float"
        # Base model
        self.feature_extractor = resnet18(pretrained=False, num_classes=4 * hidden_dim) #TODO why x4?
        self.feature_extractor.fc = Sequential(
            self.feature_extractor.fc,
            ReLU(inplace=True),
            Linear(4 * hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50) #TODO Why / 50?
        return [optimizer], [scheduler]

    def info_nce_loss(self, batch):
        imgs1, imgs2 = batch[0] #TODO refactor to remove [0]
        imgs = torch.cat([imgs1, imgs2], dim=0)

        features = self.feature_extractor(imgs)

        cos_sim = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        # Mask out self comparisons
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # For each row, mark the column of its paired true value
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # Actual loss function
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        self.log('loss', nll)

        # Move the positive samples to the front of the array for convenience
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        # For each sample, find sorted index of it's positive 
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics. (== 0) means the positive pair was sorted together.
        # The +1 is so that the metric is not 1-indexed.
        self.log("acc_top1", (sim_argsort == 0).float().mean())
        self.log("acc_top5", (sim_argsort < 5).float().mean())
        self.log("acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch)

    def predict_step(self, batch, batch_idx):
        return self.feature_extractor(batch)

from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    load_dotenv()

    #TODO unwrap pretraining set
    ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    dl = DataLoader(ds, batch_size=32)
    model = Model(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4)
    trainer = Trainer(
        gpus=1,
        max_epochs=model.hparams.max_epochs
    )
    trainer.fit(model, train_dataloader=dl)
