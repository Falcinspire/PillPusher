from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, get_label_encoder, load_label_encoder
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
import pandas as pd
import os
import argparse

from transforms import ToTensorD
from image_grid import ImageGrid

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='The csv file of prediction scores for consumer vs reference images')
    parser.add_argument('--output', type=str, required=True, help='The image file to generate')
    return parser.parse_args()

if __name__ == '__main__':
    load_dotenv()
    args = parse_arguments()

    label_encoder = load_label_encoder(os.getenv('EPILLID_LABEL_ENCODER'))

    collection = EPillIDCollection(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, filter='reference')

    prediction_df = pd.read_csv(args.source)
    relation_matrix = prediction_df[prediction_df.columns[2:]].to_numpy()
    argsorted_matrix = np.argsort(-relation_matrix, axis=1)

    consumer_imgs = prediction_df['img_path'].tolist()
    reference_imgs = [collection.get_random_entry_with_label_id(int(label_id), is_reference=True).image_path for label_id in prediction_df.columns[2:]] 
    
    create_comparison_collage(
        args.output, 
        argsorted_matrix, 
        consumer_imgs, 
        reference_imgs, 
        f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data',
        rows=64,
        columns=64,
        patch_size=64,
    )