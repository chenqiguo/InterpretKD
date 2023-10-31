from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
#from utils import *
#from util.datasets import build_transform

# newly added by Chenqi: for ImageNet-LT and CIFAR datasets.

"""
# Data transformation with augmentation
data_transforms_ImageNet = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


data_transforms_CIFAR = {
    'train': transforms.Compose([
        #transforms.Resize(224),
        transforms.RandomCrop(32, padding=4), #padding=4
        #transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'val': transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'test': transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}
"""




# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path





# Load datasets
def load_data(data_root, dataset, transform, phase): #, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True
    
    # newly added by Chenqi:
    if dataset == 'ImageNet-LT':
        # Note: data_root=args.data_path='/home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC'
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_LT/ImageNet_LT_train.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_LT/ImageNet_LT_val.txt'
        #transform = data_transforms_ImageNet[phase]
    
    elif dataset == 'CIFAR-10 imb100':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_train_imb100.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    elif dataset == 'CIFAR-10 imb50':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_train_imb50.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    elif dataset == 'CIFAR-10 imb10':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_train_imb10.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/CIFAR10_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    elif dataset == 'CIFAR-100 imb100':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_train_imb100.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    elif dataset == 'CIFAR-100 imb50':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_train_imb50.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    elif dataset == 'CIFAR-100 imb10':
        if phase == 'train':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_train_imb10.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_val.txt'
        #transform = data_transforms_CIFAR[phase]
    
    print('Loading data from %s' % (txt))

    #if phase not in ['train', 'val']:
    #    transform = data_transforms['test']
    #else:
    #    transform = data_transforms[phase]

    print('Use data transformation:', transform)

    #if shot_phase is not None:
    #    set_ = Shot_Dataset(data_root, txt, transform, shot_phase=shot_phase)
    #elif curric is True:
    #    set_ = Curric_Dataset(data_root, txt, transform)
    #else:
    set_ = LT_Dataset(data_root, txt, transform)

    return set_

    """
    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    """
    
    
