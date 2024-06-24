# SCDNet
This is the respository for our SCDNet.

## Requirements

- Python 3.8.17
- PyTorch 1.9.0
- cuda 11.1

## Preparing Few-Shot Segmentation Datasets
Download datasets following the HSNet https://github.com/juhongm999/hsnet:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data).
> Download PASCAL VOC2012 SDS extended mask annotations.

> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations:.
> Download COCO2014 train/val annotations.

## Prepare backbones

Downloading the following pre-trained backbones:

> 1. [ResNet-50](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h-35c100f8.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 2. [ResNet-101](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)

Create a directory 'backbones' to place the above backbones. The overall directory structure should be like this:

    ../                         # parent directory
    ├── SCDNet/                  # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSS dataset
    │   ├── model/              # (dir.) implementation of SCDNet
    │   ├── scripts/            # (dir.) Scripts for training and testing
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training
    │   └── test.py             # code for testing
    ├── datasets/               # (dir.) Few-Shot Segmentation Datasets
    └── backbones/              # (dir.) Pre-trained backbones

## Training
You can use our scripts to build your own. For more information, please refer to ./common/config.py

> ```bash
> sh ./scripts/train.sh
> ```
> 
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 


