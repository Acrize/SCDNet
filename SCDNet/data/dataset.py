r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import cv2
import random
import torch

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from util import dataset as BAMDataset

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, base_data_root=None):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.base_data_root = base_data_root
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, sampler=True, pin_memory=True):
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                          transform=cls.transform,
                                          split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        cls.dataset = dataset
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        
        
        # mode = 'train' if split == 'trn' else 'val'
        # dataset = BAMDataset.SemData(split=fold, shot=shot, data_root=cls.datapath, base_data_root=cls.base_data_root, \
        #                             transform=cls.transform, mode=mode, \
        #                             data_set=benchmark, use_split_coco=False)

        
        train_sampler = Sampler(dataset) if (split == 'trn' and sampler) else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
                                pin_memory=pin_memory)
        
        # train_sampler = None
        # dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
        #                         pin_memory=False)
        
        # print(f"{dataloader.dataset.mode = } {dataloader.dataset.class_ids = }")

        return dataloader