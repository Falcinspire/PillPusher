import os
from os import path
import random
from cv2 import transform
from numpy import expand_dims
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from torchvision.transforms.functional import resize
from transforms import ResizeD, RandomRotateD, RandomResizeD, ColorJitterMaskedD, RandomOverlayD, ToTensorD
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

def get_label_encoder(epillid_root):
    folds_all_path = path.join(
        epillid_root, 
        'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'
    )
    labels = pd.read_csv(folds_all_path)['label'].tolist()
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

class EPillIDCollection():
    '''
    An abstraction over the EPillID dataset for the more task-targeted datasets
    '''
    def __init__(self, epillid_root, label_encoder, use_reference_set, fold=None, use_validation=None):
        assert not (use_reference_set and fold != None), 'reference set does not have folds'
        assert not (use_reference_set and use_validation != None), 'reference set does not have validation split'
        assert not (not use_reference_set and use_validation != None and fold not in range(5)), 'consumer set with validation split must have a fold in range [0,4]'

        self.epillid_root = epillid_root
        self.epillid_data_root = path.join(epillid_root, 'classification_data')

        folds_root = path.join(epillid_root, 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/')
        if fold:
            self.df = \
                pd.read_csv(path.join(folds_root, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{fold}.csv')) \
                if use_validation else \
                pd.concat([pd.read_csv(path.join(folds_root, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{fold}.csv')) for i in range(5) if i != fold])
        else:
            self.df = pd.read_csv(path.join(folds_root, 'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'))
            self.df = self.df[self.df['is_ref'] == use_reference_set]

        # Only use front sides of images
        self.df = self.df[self.df['is_front'] == True]

        self.df['label_id'] = label_encoder.transform(self.df['label'].tolist())
        self.df = self.df.sort_values(by='label_id')

        self.df = self.df[:32] #TODO remove
    
    def get_entry(self, idx):
        return self.df.iloc[idx]

    def load_image(self, entry):
        return Image.open(path.join(self.epillid_data_root, entry['image_path']))

    def get_first_entry_with_label_id(self, label_id):
        return self.df[self.df['label_id'] == label_id].iloc[0]

    def __len__(self):
        return len(self.df)

class EPillIDSingleTypeDataset(Dataset):
    def __init__(self, epillid_root, label_encoder, use_reference_set, fold=None, use_validation=None, transforms=None):
        self.epillid_root = epillid_root
        self.transforms = transforms

        self.collection = EPillIDCollection(epillid_root, label_encoder, use_reference_set=use_reference_set, fold=fold, use_validation=use_validation)

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        entry = self.collection.get_entry(idx)
        img = self.collection.load_image(entry)

        if self.transforms != None:
            img = self.transforms(img)

        return {
            'image': img,
            'label_id': entry['label_id']
        }

    def _plot_first_images(self):
        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(wspace=0, hspace=0)
        entries = 6 # Must be even
        for idx, img in enumerate([self[idx] for idx in range(entries)]):
            plt.subplot(entries//2, 2, idx+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.permute(1, 2, 0))
        plt.show()

class EPillIDSupervisedContrastiveDataset(Dataset):
    def __init__(self, epillid_root, label_encoder, fold=0, use_validation=True, transforms=None, seed=2784958):
        assert fold in range(5)

        self.epillid_root = epillid_root
        self.transforms = transforms

        self.consumer_collection = EPillIDCollection(epillid_root, label_encoder, use_reference_set=False, fold=fold, use_validation=use_validation)
        self.reference_collection = EPillIDCollection(epillid_root, label_encoder, use_reference_set=True)

    def __len__(self):
        return len(self.consumer_collection)

    def __getitem__(self, idx):
        consumer_entry = self.consumer_collection.get_entry(idx)
        reference_entry = self.reference_collection.get_first_entry_with_label_id(consumer_entry['label_id'])

        consumer = self.consumer_collection.load_image(consumer_entry)
        reference = self.reference_collection.load_image(reference_entry)

        if self.transforms != None:
            transformed = self.transforms({ 'consumer': consumer, 'reference': reference })
            consumer = transformed['consumer']
            reference = transformed['reference']

        return {
            'positive_1': consumer,
            'positive_2': reference,
            'label_id': consumer['label_id'],
        }

    def _plot_first_images(self):
        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(wspace=0, hspace=0)
        rows = 6
        for idx, (first, second) in enumerate([self[idx] for idx in range(rows)]):
            plt.subplot(rows, 2, idx*2+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(first.permute(1, 2, 0))
            plt.subplot(rows, 2, idx*2+2)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(second.permute(1, 2, 0))
        plt.show()

if __name__ == '__main__':
    load_dotenv()

    root_dir = os.getenv('EPILLID_DATASET_ROOT')
    ds = EPillIDSupervisedContrastiveDataset(root_dir, get_label_encoder(root_dir), transforms=ToTensorD(keys=['consumer', 'reference']))
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), use_reference_set=False, fold=0, use_validation=False, transforms=ToTensor())
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), use_reference_set=False, transforms=ToTensor())
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), use_reference_set=True, transforms=ToTensor())

    ds._plot_first_images()