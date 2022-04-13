import os
from os import path
import random
from numpy import expand_dims
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from torchvision.transforms.functional import resize
from transforms import ResizeD, RandomRotateD, RandomResizeD, ColorJitterMaskedD, RandomOverlayD
import numpy as np
from dotenv import load_dotenv

class C3PIDTDMixSelfSupervisedContrastiveDataset(Dataset):
    def __init__(self, dtd_root, c3pi_reference_root, transforms=None, seed=2784958):
        self.dtd_root = dtd_root
        self.c3pi_reference_root = c3pi_reference_root
        self.transforms = \
            transforms if transforms != None \
            else Compose([
                ResizeD((224, 224), keys=['texture_image']),
                RandomRotateD(180, expand=True, keys=['pill_image']),
                RandomResizeD(size_range_longer_edge=(100, 224), keys=['pill_image']),
                ColorJitterMaskedD(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, keys=['pill_image']),
                RandomOverlayD(key_foreground='pill_image', key_background='texture_image'),
                ToTensor(),
            ])

        dtd_imdb = scipy.io.loadmat(path.join(self.dtd_root, 'imdb', 'imdb.mat'))
        dtd_images = dtd_imdb['images']['name'][0][0][0]
        # dtd_img_root = dtd_imdb['imageDir'] /data/dtd/images
        self.dtd_data = [path.join('images', dtd_image_local[0]) for dtd_image_local in dtd_images]
        random.shuffle(self.dtd_data)

        reference_data_df = pd.read_csv(path.join(c3pi_reference_root, 'index.csv'))
        reference_data_grouped_df = reference_data_df.groupby('NDC11')
        self.c3pi_reference_data = [group['Local_Path'].tolist()[0] for _, group in reference_data_grouped_df]
        random.shuffle(self.c3pi_reference_data)

    def __len__(self):
        return len(self.c3pi_reference_data)

    def __getitem__(self, idx):
        #TODO  See if transforms can be integrated into here
        pill_path = self.c3pi_reference_data[idx]
        texture_1_path = self.dtd_data[(2*idx) % len(self.dtd_data)]
        texture_2_path = self.dtd_data[(2*idx+1) % len(self.dtd_data)]

        pill = Image.open(path.join(self.c3pi_reference_root, pill_path))
        texture_1 = Image.open(path.join(self.dtd_root, texture_1_path))
        texture_2 = Image.open(path.join(self.dtd_root, texture_2_path))

        first = self.transforms({ 'pill_image': pill, 'texture_image': texture_1 })
        second = self.transforms({ 'pill_image': pill, 'texture_image': texture_2 })

        return {
            'positive_1': first,
            'positive_2': second,
            'pill_path': pill_path,
            'texture_1_path': texture_1_path,
            'texture_2_path': texture_2_path,
        }

    def _plot_random_pills(self):
        plt.figure()
        for idx, entry in enumerate(random.sample(self.c3pi_reference_data, 8)):
            plt.subplot(2, 4, idx+1)
            plt.imshow(Image.open(path.join(self.c3pi_reference_root, entry)))
        plt.show()

    def _plot_random_textures(self):
        plt.figure()
        for idx, entry in enumerate(random.sample(self.dtd_data, 8)):
            plt.subplot(2, 4, idx+1)
            plt.imshow(Image.open(path.join(self.dtd_root, entry)))
        plt.show()

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

    ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    # ds._plot_random_pills()
    # ds._plot_random_textures()

    ds._plot_first_images()