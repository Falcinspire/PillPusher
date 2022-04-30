from dotenv import load_dotenv
import numpy as np
import pandas as pd
import argparse
import json
import os
from os import path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from PIL import Image
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from epillid_datasets import EPillIDSingleTypeDataset, EPillIDSupervisedContrastiveDataset, load_label_encoder
from model import Model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

from epillid.metrics import mapk, average_precision_score, global_average_precision
from image_grid import ImageGrid

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='tsne, activation, grad-cam')
    parser.add_argument('--activation-indices', type=str)
    parser.add_argument('--source', type=str, help='The csv file of embeddings')
    parser.add_argument('--show-plot', default=False, action='store_true')
    parser.add_argument('--show-random-neighbors', default=False, action='store_true')
    parser.add_argument('--output', type=str, help='The png file to generate')
    parser.add_argument('--grad-cam-model-checkpoint', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = parse_arguments()

    if args.mode == 'tsne':
        prediction_df = pd.read_csv(args.source)
        labels = [str(i) for i in range(128)]
        classification_root = f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data'

        saved_file = f'{args.output}.npy'
        if path.isfile(saved_file):
            print('Using embeddings from last run')
            X_embedded = np.load(saved_file).reshape(-1, 2)
        else:
            X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(prediction_df[labels])
            print(X_embedded.shape, X_embedded.dtype)
            np.save(args.output, X_embedded)

        if args.show_plot:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
            plt.show()

        if args.show_random_neighbors:
            center = random.randint(0, X_embedded.shape[0]-1)
            distances = np.sqrt(np.sum(np.square(X_embedded - X_embedded[center]), axis=-1))
            nearest = np.argsort(distances)[:24]
            center_path = prediction_df.iloc[center].img_path
            nearest_paths = [entry.img_path for index, entry in prediction_df.iloc[nearest].iterrows()]

            image_grid = ImageGrid(5, 5, 128)
            for idx, entry in enumerate([center_path] + nearest_paths):
                image_grid.draw(idx // 5, idx % 5, Image.open(path.join(classification_root, entry)))
            image_grid.show()
    elif args.mode == 'activation':
        prediction_df = pd.read_csv(args.source)
        labels = [str(i) for i in range(128)]
        classification_root = f'{os.getenv("EPILLID_DATASET_ROOT")}/classification_data'

        activation_indices = [int(value) for value in args.activation_indices.split(',')]
        sorted_embeddings = np.argsort(prediction_df[labels].to_numpy(), axis=0)[:8, activation_indices]
        image_grid = ImageGrid(len(activation_indices), 8, 128)
        for act_idx, activation_index_values in enumerate(sorted_embeddings.transpose()):
            paths = [entry.img_path for _, entry in prediction_df.iloc[activation_index_values].iterrows()]
            for idx, entry in enumerate(paths):
                image_grid.draw(act_idx, idx, Image.open(path.join(classification_root, entry)))
        image_grid.show()
    elif args.mode == 'grad-cam':
        model = Model.load_from_checkpoint(args.grad_cam_model_checkpoint)
        target_layers = [
            model.feature_extractor.inception5b.branch2[1].bn,
            model.feature_extractor.inception5b.branch3[1].bn,
            model.feature_extractor.inception5b.branch4[1].bn,
        ]
        cam =  GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        dataloader = DataLoader(
            C3PIDTDMixSelfSupervisedContrastiveDataset(
                os.getenv('DTD_DATASET_ROOT'), 
                os.getenv('C3PI_REFERENCE_PILLS')
            ),
            batch_size=12,
            num_workers=1,
            shuffle=True,
        )
        rgb_images = next(iter(dataloader))['positive_1']
        grayscale_cam = cam(
            input_tensor=rgb_images.to(device='cuda'),
            targets = [ClassifierOutputTarget(0)],
        )
        for image, grayscale in zip(rgb_images, grayscale_cam):
            visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale, use_rgb=True)
            plt.imshow(visualization)
            plt.show()
