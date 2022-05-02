# PillPusher

## Training
Sample training commands:
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
