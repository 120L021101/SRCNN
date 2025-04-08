# FusionSR

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

For standalone SRCNN & Transformer training, use the following command (put both train and eval dataset into /data folder)

(for Transformer, comment line 41 & line 44-48, then uncomment line 42 & line 49-53 in train_v.py to switch model):

```bash
python train_v.py --train-file "data/91-image_x3.h5" --eval-file "data/Set5_x3.h5" --outputs-dir "data/outputs" --scale 3 --lr 1e-4 --batch-size 16 --num-epoch 100 --num-workers 8 --seed 123  
```

For models + GAN-based training, use train.py instead

(now comment line 70 & line 77-81, then uncomment line 71 & line 82-86 in train.py to switch to Transformer):

```bash
python train.py --train-file "data/91-image_x3.h5" --eval-file "data/Set5_x3.h5" --outputs-dir "data/outputs" --scale 3 --lr 1e-4 --batch-size 16 --num-epoch 100 --num-workers 8 --seed 123  
```

## Ablation Variants
### Loss weights
Weights for each loss term can be changed at line 163 in train.py. Note that the vgg loss is commented by default.
If uncommented at line 163, also uncomment its computation step at line 161.

### Label Smoothing
Comment the hard real label at line 122 and uncomment the smoothed label at line 123 in train.py to turn on label smoothing.

### WGAN
First comment line 50 in models.py (the Sigmoid function),

then comment line 143-145 and uncomment line 148-149 in train.py for WGAN dloss computation,

at last comment line 157 and uncomment line 159 for WGAN gloss computation.

## Test
Use the following command to run test on models (best.pth is save after all epochs are done)

(comment/uncomment line 22 & 23 in test.py to match the model used in training):
```bash
python test.py --weights-file "data/outputs/x3/best.pth" --image-file "data/butterfly_GT.bmp" --scale 3   
```
