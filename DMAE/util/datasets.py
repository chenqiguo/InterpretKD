# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# newly added by Chenqi:
#from util import dataloader_LT
from util import dataloader_LT_finetune

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    # newly modified by Chenqi:
    if 'ILSVRC' in args.data_path and 'LT' not in args.output_dir:
        print('--> For original (balanced) ImageNet all dataset')
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    
    if 'ILSVRC' in args.data_path and 'LT' in args.output_dir:
        print('--> For long-tail ImageNet-LT dataset')
        dataset = dataloader_LT_finetune.load_data(data_root=args.data_path, dataset='ImageNet-LT', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-10-' in args.data_path and 'imb100/' in args.output_dir:
        print('--> For CIFAR-10 imb100 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-10 imb100', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-10-' in args.data_path and 'imb50/' in args.output_dir:
        print('--> For CIFAR-10 imb50 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-10 imb50', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-10-' in args.data_path and 'imb10/' in args.output_dir:
        print('--> For CIFAR-10 imb10 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-10 imb10', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-100-' in args.data_path and 'imb100/' in args.output_dir:
        print('--> For CIFAR-100 imb100 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-100 imb100', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-100-' in args.data_path and 'imb50/' in args.output_dir:
        print('--> For CIFAR-100 imb50 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-100 imb50', transform=transform, phase='train' if is_train else 'val')
    
    if 'cifar-100-' in args.data_path and 'imb10/' in args.output_dir:
        print('--> For CIFAR-100 imb10 dataset')
        dataset = dataloader_LT_finetune.load_data(data_root='', dataset='CIFAR-100 imb10', transform=transform, phase='train' if is_train else 'val')
    
    if 'inaturalist' in args.data_path and 'all' in args.data_path:
        print('--> For original iNaturalist all dataset')
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        
    #root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    
    # newly modified by Chenqi:
    # for ImageNet:
    if 'ILSVRC' in args.data_path:
        print('--> For ImageNet dataset')
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    
    # for other datasets:
    if 'cifar' in args.data_path:
        print('--> For CIFAR dataset')
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    
    if 'inaturalist' in args.data_path and 'all' in args.data_path:
        print('--> For original iNaturalist all dataset')
        #normalize = transforms.Normalize(mean=[0.4605, 0.4786, 0.3684],
        #                                 std=[0.1862, 0.1844, 0.1826]) # from get_mean_std.py
        mean = [0.4605, 0.4786, 0.3684]
        std = [0.1862, 0.1844, 0.1826]
    
    #mean = IMAGENET_DEFAULT_MEAN
    #std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
