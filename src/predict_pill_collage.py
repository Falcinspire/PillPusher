from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, get_label_encoder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from model import Model
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from os import path
from tqdm import tqdm
import os

from transforms import ToTensorD
from image_grid import ImageGrid

#refs https://holypython.com/python-pil-tutorial/creating-photo-collages/#:~:text=An%20image%20collage%20can%20easily,to%20the%20grid%20through%20iteration.
# def create_pill_collage_rows(argsorted_matrix, consumer_output_imgs, consumer_output_lbs, reference_output_imgs, root_img_dir, patch_size=224, comparisons=16):
#     if not path.isdir('collage'):
#         os.mkdir('collage')

#     for consumer_idx, consumer_img_path in enumerate(tqdm(consumer_output_imgs)):
#         # Debugging/wanna know stuff
#         #TODO remove
#         argsorted_row = argsorted_matrix[consumer_idx]
#         print()
#         print(consumer_img_path, reference_output_imgs[consumer_output_lbs[consumer_idx]])
#         print(consumer_idx, (argsorted_row == consumer_output_lbs[consumer_idx]).nonzero().item())
#         canvas = Image.new("RGB", ((comparisons+1)*patch_size, patch_size))
#         consumer_img = Image.open(path.join(root_img_dir, consumer_img_path)).resize((patch_size, patch_size))
#         canvas.paste(consumer_img, (0, 0))
#         for reference_idx, reference_img_ipath in enumerate(argsorted_matrix[consumer_idx][:comparisons]):
#             reference_img = Image.open(path.join(root_img_dir, reference_output_imgs[reference_img_ipath])).resize((patch_size, patch_size))
#             canvas.paste(reference_img, (patch_size*(reference_idx+1), 0))
#         canvas.save(f'collage/{str(consumer_idx)}.png')

def create_comparison_collage(filename, argsorted_matrix, independent_output_imgs, dependent_output_imgs, root_img_dir, patch_size=16, rows=16, columns=16):
    if not path.isdir('collage'):
        os.mkdir('collage')
    
    collage = ImageGrid(rows, columns, patch_size)
    for consumer_idx in tqdm(range(rows)):
        sorted_row = argsorted_matrix[consumer_idx]
        collage.draw(consumer_idx, 0, Image.open(path.join(root_img_dir, independent_output_imgs[consumer_idx])))
        for reference_idx in range(columns):
            collage.draw(consumer_idx, reference_idx+1, Image.open(path.join(root_img_dir, dependent_output_imgs[sorted_row[reference_idx]])))
    collage.save(filename)

if __name__ == '__main__':
    load_dotenv()

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
    # ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    # ds = EPillIDDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder=label_encoder, transforms=ToTensorD(keys=['consumer', 'reference']))
    # dl = DataLoader(ds, batch_size=32)
    # model = Model(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4)
    model = Model.load_from_checkpoint('outputs/pretrained-googlenet-v1/pretrained-googlenet-v1.ckpt')
    trainer = Trainer(
        gpus=1,
        max_epochs=model.hparams.max_epochs,
    )
    # trainer.fit(model, train_dataloaders=[dl])
    consumer_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, fold='all', use_reference_set=False, transforms=ToTensor()),
                batch_size=32,
                shuffle=False,
                # sampler=RandomSampler(list(range(100)), num_samples=32, replacement=True),
            ),
        ],
    )
    consumer_output = torch.vstack([batch['image'] for batch in consumer_batch_output])
    #refs https://stackoverflow.com/a/952952
    consumer_output_imgs = [item for batch in consumer_batch_output for item in batch['image_path']]
    consumer_output_lbs = torch.cat([batch['label_id'] for batch in consumer_batch_output])

    reference_batch_output = trainer.predict(
        model, 
        dataloaders=[
            DataLoader(
                EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, fold='all', use_reference_set=True, transforms=ToTensor()),
                batch_size=32,
                shuffle=False,
                # sampler=RandomSampler(list(range(100)), num_samples=32, replacement=True),
            ),
        ],
    )
    reference_output = torch.vstack([batch['image'] for batch in reference_batch_output])
    reference_output_imgs = [item for batch in reference_batch_output for item in batch['image_path']]
    reference_output_lbs = torch.cat([batch['label_id'] for batch in reference_batch_output])
    print('reference_output_lbs', len(reference_output_lbs))
    reference_output_real_labels = label_encoder.inverse_transform(reference_output_lbs.numpy())

    # Similarity comparison
    cos_sim = F.cosine_similarity(consumer_output[:64, None], reference_output[:], dim=-1)
    sorted_sim = torch.argsort(cos_sim, dim=1, descending=True)
    create_comparison_collage(
        f'collage/collage_mixed_{str(64)}x{str(64)}_{64}.png',
        sorted_sim, 
        consumer_output_imgs, 
        reference_output_imgs, 
        f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data',
        rows=64,
        columns=64,
        patch_size=64,
    )

    cos_sim = F.cosine_similarity(consumer_output[:64, None], consumer_output[:64], dim=-1)
    sorted_sim = torch.argsort(cos_sim, dim=1, descending=True)
    create_comparison_collage(
        f'collage/collage_cons_{str(64)}x{str(64)}_{64}.png',
        sorted_sim, 
        consumer_output_imgs, 
        consumer_output_imgs, 
        f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data',
        rows=64,
        columns=64,
        patch_size=64,
    )

    cos_sim = F.cosine_similarity(reference_output[:64, None], reference_output[:64], dim=-1)
    sorted_sim = torch.argsort(cos_sim, dim=1, descending=True)
    create_comparison_collage(
        f'collage/collage_ref_{str(64)}x{str(64)}_{64}.png',
        sorted_sim, 
        reference_output_imgs, 
        reference_output_imgs, 
        f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data',
        rows=64,
        columns=64,
        patch_size=64,
    )

    # pos_mask = torch.zeros((consumer_output.shape[0], reference_output.shape[0]), dtype=torch.bool)
    # for idx, label in enumerate(consumer_output_lbs):
    #     pos_mask[idx, label] = True

    # # Move the positive samples to the front of the array for convenience
    # comb_sim = torch.cat(
    #     [cos_sim[pos_mask][:, None], cos_sim],  # Correct reference image
    #     dim=-1,
    # )

    # #TODO add metadata to datasets
    # #TODO metric calculation