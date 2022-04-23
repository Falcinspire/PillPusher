from dotenv import load_dotenv
from c3pi_dtd_mix_selfsupervised_dataset import C3PIDTDMixSelfSupervisedContrastiveDataset
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, Compose, ColorJitter, ToTensor
from epillid_datasets import EPillIDCollection, EPillIDSingleTypeDataset, get_label_encoder
import os

if __name__ == '__main__':
    load_dotenv()

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
   
    all_epillid_ndc11 = EPillIDSingleTypeDataset(os.getenv('EPILLID_DATASET_ROOT'), label_encoder, fold='all', use_reference_set=True).get_unique_ndc11()
    all_c3pi_ndc11 = C3PIDTDMixSelfSupervisedContrastiveDataset(os.getenv('DTD_DATASET_ROOT'), os.getenv('C3PI_REFERENCE_PILLS')).get_unique_ndc11s()
    print('shared', len(all_epillid_ndc11.intersection(all_c3pi_ndc11)))
    print('total epillid labels', len(all_epillid_ndc11))
    # print('total unshared', len(all_epillid_ndc11) + len(all_c3pi_ndc11))
    # print('total unique', len(all_epillid_ndc11.union(all_c3pi_ndc11)))
    print('percent of set already seen', 100 * len(all_epillid_ndc11.intersection(all_c3pi_ndc11))/len(all_epillid_ndc11))