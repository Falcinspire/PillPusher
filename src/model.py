#refs https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/13-contrastive-learning.html

from sched import scheduler
from torchvision.models import resnet18
import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ToPILImage
from pytorch_lightning import LightningModule, Trainer
import numpy as np

import matplotlib.pyplot as plt

from image_grid import ImageGrid

class Model(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, pretraining=False, fold=0, batch_size=256, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float"
        # Base model
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        self.feature_extractor.fc = Sequential(
            self.feature_extractor.fc,
            ReLU(inplace=True),
            Linear(self.feature_extractor.fc.out_features, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50) #TODO Why / 50?
        return [optimizer], [scheduler]

    def info_nce_loss(self, batch, dump=False):
        imgs1, imgs2 = batch['positive_1'], batch['positive_2'] #TODO remove the [0]. Maybe use a collate_fn?
        imgs = torch.cat([imgs1, imgs2], dim=0)

        features = self.feature_extractor(imgs)

        cos_sim = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)

        # Testing output
        #TODO the conversion from tensor to PIL to np is not great
        if dump:
            raw_sorted = cos_sim.argsort(dim=-1, descending=True)
            rows = min(32, imgs.shape[0])
            base_cols = min(16, imgs.shape[0])
            image_grid = ImageGrid(rows, base_cols+1, 128)
            for row, row_value in enumerate(raw_sorted[:rows]):
                image_grid.draw(row, 0, ToPILImage()(imgs[row]))
                for col, col_value in enumerate(row_value[:base_cols]):
                    image_grid.draw(row, 1+col, ToPILImage()(imgs[col_value]))
            #TODO get current epoch
            self.logger.experiment.add_image(f'validation_batch_{self.current_epoch}', np.asarray(image_grid.canvas).transpose((2, 0, 1)), 0) 

        # Mask out self comparisons
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # For each row, mark the column of its paired true value
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # Actual loss function
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Move the positive samples to the front of the array for convenience
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        # For each sample, find sorted index of it's positive 
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics. (== 0) means the positive pair was sorted together.
        # The +1 is so that the metric is not 1-indexed.
        acc_top1 = (sim_argsort == 0).float().mean()
        acc_top5 = (sim_argsort < 5).float().mean()
        acc_mean_pos = 1 + sim_argsort.float().mean()

        return nll, acc_top1, acc_top5, acc_mean_pos

    def training_step(self, batch, batch_idx):
        if (type(batch) == list): batch = batch[0] #TODO I don't know why this happens
        nll, acc_top1, acc_top5, acc_mean_pos = self.info_nce_loss(batch)

        self.log('training_loss', nll, batch_size=self.hparams.batch_size)
        self.log('training_acc_top1', acc_top1, batch_size=self.hparams.batch_size)
        self.log('training_acc_top5', acc_top5, batch_size=self.hparams.batch_size)
        self.log('training_acc_mean_pos', acc_mean_pos, batch_size=self.hparams.batch_size)

        return nll

    def training_step_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        nll, acc_top1, acc_top5, acc_mean_pos = self.info_nce_loss(batch, dump=(batch_idx == 0))

        self.log('validation_loss', nll, batch_size=self.hparams.batch_size)
        self.log('validation_acc_top1', acc_top1, batch_size=self.hparams.batch_size)
        self.log('validation_acc_top5', acc_top5, batch_size=self.hparams.batch_size)
        self.log('validation_acc_mean_pos', acc_mean_pos, batch_size=self.hparams.batch_size)

        return nll

    def predict_step(self, batch, batch_idx):
        imgs = batch['image']
        return_value = batch.copy()
        return_value['image'] = self.feature_extractor(imgs)
        return return_value

from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    load_dotenv()

    #TODO unwrap pretraining set
    ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    dl = DataLoader(ds, batch_size=32)
    dl_validation = DataLoader(ds, batch_size=32)
    model = Model(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4)
    trainer = Trainer(
        gpus=1,
        max_epochs=model.hparams.max_epochs
    )
    trainer.fit(
        model, 
        train_dataloaders=[dl], 
        validation_dataloaders=[dl_validation],
        limit_val_batches=1,
    )
