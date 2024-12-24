# LIPIDS
**[LIPIDS: Learning-based Illumination Planning In Discretized(Light) Space for Photomertic Stereo](https://arxiv.org/abs/2409.02716)**,


This paper addresses the problem of finding best light directions for photometric stereo backbones in discretized light space.



## Dependencies
We are using NRNet inspired from PS-FCN is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction. 
- Python 3.7 
- PyTorch (version = 1.10)
- numpy
- scipy
- CUDA-9.0  

## Overview
We provide:
- Downloading Datasets
- Code for training your own LIPIDS.
- Get the best configuration among given finite options.

## Prepare Dataset for training
We are using Blobby and Sculpture dataset to train LIPIDS. You can download it from [PS_FCN](https://github.com/guanyingc/PS-FCN). Place data under `./data/datasets/`.

## Training
Run below command to train your model.
```
CUDA_VISIBLE_DEVICES=0 python main.py --concat_data --dataset PS_SampleNet_Dataset --model LSNET --out_img 10 --item out10 --lr_decay 1.0
```
- `out_img`: you can use this to train model to choose 10 light directions out of 48.
- `item`: folder you want to store trained model. (eg. `./data/models/Training/out10`)

## Get the best configuration
To get best light configuration after training. Run the below command and it will save the light plot as `lighting.png` file

```
python get_configs.py {path to your trained model}
```

## Citation
If you find this project useful please cite:

Tiwari Ashish, Sutariya Mihir, and Shanmuganathan Raman. 2024. "LIPIDS: Learning-based Illumination Planning In Discretized (Light) Space for Photometric Stereo." arXiv. https://doi.org/10.48550/arXiv.2409.02716
