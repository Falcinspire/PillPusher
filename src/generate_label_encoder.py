import pickle
from dotenv import load_dotenv
import argparse
import os
from epillid_datasets import get_label_encoder, save_label_encoder

if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True)
    args = parser.parse_args()

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
    save_label_encoder(label_encoder, args.filename)