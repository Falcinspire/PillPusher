import asyncio
import random
import argparse
import os
from os import path
from PIL import Image, ImageOps
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from scipy import ndimage
from dotenv import load_dotenv

load_dotenv()

def generate_image(local_url):
    response = requests.get(f'https://data.lhncbc.nlm.nih.gov/public/Pills/{local_url}')
    image1 = Image.open(BytesIO(response.content))
    
    segmentation_zone = image1.crop((0, 0, 2400, 1600))
    cropped_pill = segmentation_zone.getbbox()

    return segmentation_zone.crop(cropped_pill)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data')
    args = parser.parse_args()

    if not path.exists(args.output):
        os.makedirs(args.output)

    df = pd.read_csv(os.getenv('C3PI_DATASET_CSV'))
    df = df[df['Layout'] == 'MC_C3PI_REFERENCE_SEG_V1.6']
    df['Local_Path'] = pd.NaT

    #refs https://stackabuse.com/how-to-iterate-over-rows-in-a-pandas-dataframe/, https://stackoverflow.com/a/45716191
    for step, entry in enumerate(df.itertuples()):
        if step > 0 and step % 100 == 0:
            df.to_csv(f'{args.output}/index.csv', index=False)

        img_name = path.basename(entry.Image)
        df.loc[entry.Index, 'Local_Path'] = img_name
        if path.exists(f'{args.output}/{img_name}'):
            print(f'Entry {entry.Image} already exists, skipping download')
            continue
        print(f'Generating for {entry.Image}...')
        image = generate_image(entry.Image)
        image.save(f'{args.output}/{img_name}')

    df.to_csv(f'{args.output}/index.csv', index=False)