import os
from os import path
import random
from numpy import expand_dims
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, RandomAffine, RandomPerspective, RandomHorizontalFlip, ToTensor, ToPILImage
from torchvision.transforms.functional import resize
from image_grid import ImageGrid
from transforms import ResizeD, RandomRotateD, RandomResizeD, ColorJitterMaskedD, RandomOverlayD, TransformD
import numpy as np
from dotenv import load_dotenv

class C3PIDTDMixSelfSupervisedContrastiveDataset(Dataset):
    def __init__(self, dtd_root, c3pi_reference_root, transforms=None):
        self.dtd_root = dtd_root
        self.c3pi_reference_root = c3pi_reference_root
        self.transforms = \
            transforms if transforms != None \
            else Compose([
                ResizeD((224, 224), keys=['texture_image']),
                RandomResizeD(size_range_longer_edge=(512, 512), keys=['pill_image']), # reduce size for performance reasons
                RandomRotateD(180, expand=True, keys=['pill_image']),
                TransformD(transform=RandomHorizontalFlip(), keys=['pill_image']),
                TransformD(transform=RandomPerspective(
                    distortion_scale=0.6
                ), keys=['pill_image']),
                ColorJitterMaskedD(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, keys=['pill_image']),
                RandomResizeD(size_range_longer_edge=(200, 224), keys=['pill_image']),
                RandomOverlayD(key_foreground='pill_image', key_background='texture_image'),
                ToTensor(),
            ])

        dtd_imdb = scipy.io.loadmat(path.join(self.dtd_root, 'imdb', 'imdb.mat'))
        dtd_images = dtd_imdb['images']['name'][0][0][0]
        # dtd_img_root = dtd_imdb['imageDir'] /data/dtd/images
        self.dtd_data = [path.join('images', dtd_image_local[0]) for dtd_image_local in dtd_images]
        random.shuffle(self.dtd_data)

        # No longer care about grouping by front/back
        # reference_data_df = pd.read_csv(path.join(c3pi_reference_root, 'index.csv'))
        # reference_data_grouped_df = reference_data_df.groupby('NDC11')
        # _c3pi_reference_data = [group['Local_Path'].tolist() for _, group in reference_data_grouped_df]
        # print(len(_c3pi_reference_data))
        # random.shuffle(self.c3pi_reference_data)
        # freq = {}
        # for list in _c3pi_reference_data:
        #     if (len(list)) == 32: print(list)
        #     if len(list) not in freq:
        #         freq[len(list)] = 1
        #     else:
        #         freq[len(list)] += 1
        # print(freq)

        self.c3pi_reference_data_index = pd.read_csv(path.join(c3pi_reference_root, 'index.csv'))
        self.c3pi_reference_data = self.c3pi_reference_data_index['Local_Path'].tolist()
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

    def get_unique_ndc11s(self):
        return set([f'{value:011d}' for value in self.c3pi_reference_data_index['NDC11'].tolist()])

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

    def _plot_first_images(self, square_count=16):
        iterator = iter(self)
        image_grid = ImageGrid(square_count, square_count, 224)
        toPILTransform = ToPILImage()
        for i in range(square_count):
            for j in range(square_count//2):
                output = next(iterator)
                image_grid.draw(i, j*2, toPILTransform(output['positive_1']))
                image_grid.draw(i, j*2+1, toPILTransform(output['positive_2']))
        image_grid.show()

if __name__ == '__main__':
    load_dotenv()

    ds = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS'))
    # ds._plot_random_pills()
    # ds._plot_random_textures()

    ds._plot_first_images()