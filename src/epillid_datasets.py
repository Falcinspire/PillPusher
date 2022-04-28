import os
from os import path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor, ToPILImage, RandomPerspective, RandomHorizontalFlip
from torchvision.transforms.functional import resize
from image_grid import ImageGrid
from transforms import NewPad, ResizeD, RandomRotateD, RandomResizeD, ColorJitterMaskedD, RandomOverlayD, ToTensorD, TransformD
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
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

def save_label_encoder(encoder, filepath):
    with open(filepath, 'wb+') as out:
        pickle.dump(encoder.classes_, out)

def load_label_encoder(filepath):
    label_encoder = LabelEncoder()
    with open(filepath, 'rb') as inp:
        label_encoder.classes_ = pickle.load(inp)
    return label_encoder

class EPillIDCollection():
    '''
    An abstraction over the EPillID dataset for the more task-targeted datasets

    EPillIDCollection(fold=0, validation=False, inject_references=True): training set for first fold
    EPillIDCollection(fold=0, validation=True, inject_references=True): validation set for first fold
    EPillIDCollection(filter='consumer'): entire consumer set
    EPillIDCollection(filter='reference'): entire reference set
    EPillIDCollection(): entire set
    '''
    def __init__(self, epillid_root, label_encoder, fold=None, validation=False, filter=None, inject_references=False, sort_by_label_id=False):
        self.epillid_root = epillid_root
        self.epillid_data_root = path.join(epillid_root, 'classification_data')

        folds_root = path.join(epillid_root, 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/')

        if fold != None:
            if validation:
                self.df = pd.read_csv(path.join(folds_root, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{fold}.csv'))
            else:
                self.df = pd.concat([pd.read_csv(path.join(folds_root, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{i}.csv')) for i in range(5) if i != fold])
        else:
            self.df = pd.read_csv(path.join(folds_root, 'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'))

        if filter != None:
            self.df = self.df[self.df['is_ref'] == (filter == 'reference')]

        if inject_references:
            all_folds_df = pd.read_csv(path.join(folds_root, f'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'))
            all_folds_df = all_folds_df[all_folds_df['is_ref'] == True]
            self.df = pd.concat([self.df, all_folds_df])

        # Only use front sides of images
        self.df = self.df[self.df['is_front'] == True]

        self.df['label_id'] = label_encoder.transform(self.df['label'].tolist())

        self.df = \
            self.df.sort_values(by=['label_id']).reset_index(drop=True) \
            if sort_by_label_id else \
            self.df.sample(frac=1).reset_index(drop=True)
    
    def get_entry(self, idx):
        return self.df.iloc[idx]

    def load_image(self, entry):
        return Image.open(path.join(self.epillid_data_root, entry['image_path']))

    def get_random_entry_with_label_id(self, label_id, is_reference):
        return self.df[(self.df['is_ref'] == is_reference) & (self.df['label_id'] == label_id)].sample().iloc[0]

    def _filter_unknown_labels(self, labels):
        return [label for label in labels if label.find('_') != -1]

    def _format_ndc11(self, label):
        label = label[:label.index('_')]
        label = label.replace('-', '')
        label = int(label)
        return f'{label:011d}'

    def get_unique_ndc11(self):
        return set([self._format_ndc11(label) for label in self._filter_unknown_labels(self.df['pilltype_id'].tolist())])

    def __len__(self):
        return len(self.df)

class EPillIDSingleTypeDataset(Dataset):
    def __init__(self, epillid_root, label_encoder, use_reference, fold=None, validation=None, transforms=None):
        self.epillid_root = epillid_root
        self.transforms = \
            transforms if transforms != None \
            else ToTensor()

        self.collection = EPillIDCollection(
            epillid_root, 
            label_encoder,
            fold=fold,
            validation=None,
            filter='reference' if use_reference else 'consumer',
            sort_by_label_id=True,
        )

    def get_unique_ndc11(self):
        return self.collection.get_unique_ndc11()

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        entry = self.collection.get_entry(idx)
        img = self.collection.load_image(entry)

        if self.transforms != None:
            img = self.transforms(img)

        return {
            'image': img,
            'label': entry['label'],
            'label_id': np.array(entry['label_id']),
            'image_path': entry['image_path'],
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
    def __init__(self, epillid_root, label_encoder, fold=None, validation=False, transforms=None):
        assert fold == None or fold in range(5)

        self.epillid_root = epillid_root
        self.transforms = \
            transforms if transforms != None \
            else Compose([
                TransformD(transform=Resize((512, 512)), keys=['positive_1', 'positive_2']), # reduce size for performance reasons
                TransformD(transform=RandomRotation(180, expand=True), keys=['positive_1', 'positive_2']),
                TransformD(transform=RandomHorizontalFlip(), keys=['positive_1', 'positive_2']),
                TransformD(transform=RandomPerspective(
                    distortion_scale=0.6,
                ), keys=['positive_1', 'positive_2']),
                TransformD(transform=ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2,
                ), keys=['positive_1', 'positive_2']),
                RandomResizeD(size_range_longer_edge=(220, 224), keys=['positive_1', 'positive_2']),
                TransformD(transform=NewPad((224, 224)), keys=['positive_1', 'positive_2']),
                TransformD(transform=ToTensor(), keys=['positive_1', 'positive_2']),
            ])

        self.collection = EPillIDCollection(
            epillid_root,
            label_encoder,
            fold=fold,
            validation=validation,
            inject_references=True,
        )

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        first_entry = self.collection.get_entry(idx)
        second_entry = None

        if first_entry['is_ref']:
            second_entry = first_entry
        else:
            second_entry = self.collection.get_random_entry_with_label_id(first_entry['label_id'], is_reference=True)

        first = self.collection.load_image(first_entry)
        second = self.collection.load_image(second_entry)

        if self.transforms != None:
            transformed = self.transforms({ 'positive_1': first, 'positive_2': second })
            first = transformed['positive_1']
            second = transformed['positive_2']

        return {
            'positive_1': first,
            'positive_1_path': first_entry['image_path'],
            'positive_2': second,
            'positive_2_path': second_entry['image_path'],
            'label': first_entry['label'],
            'label_id': np.array(first_entry['label_id']),
        }

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

    root_dir = os.getenv('EPILLID_DATASET_ROOT')
    ds = EPillIDSupervisedContrastiveDataset(root_dir, load_label_encoder(os.getenv('EPILLID_LABEL_ENCODER')), fold=1, validation=True)
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), fold=0, use_reference=False)
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), use_reference_set=False, transforms=ToTensor())
    # ds = EPillIDSingleTypeDataset(root_dir, get_label_encoder(root_dir), use_reference_set=True, transforms=ToTensor())

    ds._plot_first_images()

    # for entry in ds:
    #     pass