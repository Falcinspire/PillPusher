# PillPusher

# Installation

1. There are a number of dependencies that must be installed. See the imports for each file for details.

2. There must be a .env file with a few environment variables:
```sh
EPILLID_DATASET_ROOT=<root of the ePillID benchmark data>
EPILLID_LABEL_ENCODER=<file where the label encoder is stored>
C3PI_DATASET_CSV=<location of the c3pi primary csv. This can be downloaded from https://www.nlm.nih.gov/databases/download/pill_image.html>
DTD_DATASET_ROOT=<root of the Describable Textures Dataset data>
C3PI_REFERENCE_PILLS=<root of the C3PI reference data>
```
Some of these need to be filled later. 


3. Run the script to download the C3PI reference data locally:
```python generate_c3pi_ref_dataset.py --output ./data```
4. Run the script to generate the label encoder data:
```python generate_label_encoder.py --filename label_encoder.pickle```
5. Update the .env file per results from (3) and (4)

## Training Sample
```sh
python train.py \
--fold 0 \
--checkpoint-path /checkpoints/fold0 \
--checkpoint-name fold0 \
--epochs 200 \
--gpus 1 \
--dataloader-num-workers 2 \
--batch-size 128 \
```

```sh
python train.py \
--fold 0 \
--checkpoint-path checkpoints/fold0-pretrained \
--checkpoint-name fold0-pretrained \
--epochs 200 \
--gpus 1 \
--dataloader-num-workers 2 \
--batch-size 128 \
--pretrain-from-checkpoint checkpoitns/pretrained/pretrained.ckpt \
```

```sh
python train.py \
--pretraining \
--checkpoint-path checkpoints/pretrained \
--checkpoint-name pretrained \
--epochs 200 \
--gpus 1 \
--dataloader-num-workers 2 \
--batch-size 128 \
```

## Evaluation Sample
1. Generate csv relating consumer to reference images:

`python predict_for_eval.py --batch-size 32 --fold 0 --validation --checkpoint checkpoints/fold0/fold0.ckpt --output fold0-f0`

2. Calculate evaluation metrics on the file

`python evaluate.py --source fold0-f0.csv --output fold0-f0-eval`

## Embeddings Visualization
### Grad-CAM
`python embedding_space.py --mode grad-cam --source embeddings.csv --grad-cam-model-checkpoint-1 checkpoints/fold1/fold1.ckpt --grad-cam-model-checkpoint-2 checkpoints/fold1-pretrained/fold1-pretrained.ckpt --save-grad-cam grad_cam.png`

### t-SME
1. Generate the raw embeddings output

`python embeddings.py --batch-size 32 --checkpoint checkpoints/fold0-pretrained/fold0-pretrained.ckpt --output embeddings.csv`

2. Run the t-SME command(s) desired. The `--output` flags are used to cache the t-SNE results.

`python embedding_space.py --mode tsne --source embeddings.csv --save-plot tsne_plot.png --output embeddings`
`python embedding_space.py --mode tsne --source embeddings.csv --save-random-neighbors tsne_rn.png --output embeddings`

### Activations
1. Generate the raw embeddings output

`python embeddings.py --batch-size 32 --checkpoint checkpoints/fold0-pretrained/fold0-pretrained.ckpt --output embeddings.csv`

2. Run the activations command

`python embedding_space.py --mode activation --source embeddings.csv --activation-indices 11,12,13,14 --save-activation activation11121314.png`
