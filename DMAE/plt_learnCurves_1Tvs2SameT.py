#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:01:38 2023

@author: ps
"""

# plot the learning curves for comparing the results of:
# 1 Teacher vs. 2 Same Teachers
# in each long-tail dataset case.

import os
import matplotlib.pyplot as plt
import json

srcRootDir = '/home/ps/scratch/KD_imbalance/DMAE/work_dirs/'

folder_list = [('distill_finetuned_base_model_ImageNetLT_e1/','distill_finetuned_base_model_ImageNetLT_e2s1_2sameT/'),
               ('distill_finetuned_base_model_inaturalist_e1/','distill_finetuned_base_model_inaturalist_e2s1_2sameT/'),
               ('distill_finetuned_base_model_CIFAR10_e3/imb100/','distill_finetuned_base_model_CIFAR10_e6s1_2sameT/imb100/'),
               ('distill_finetuned_base_model_CIFAR10_e3/imb10/','distill_finetuned_base_model_CIFAR10_e6s1_2sameT/imb10/'),
               ('distill_finetuned_base_model_CIFAR100_e3/imb100/','distill_finetuned_base_model_CIFAR100_e4s1_2sameT/imb100/'),
               ('distill_finetuned_base_model_CIFAR100_e3/imb10/','distill_finetuned_base_model_CIFAR100_e4s1_2sameT/imb10/')]

txtFile = 'log.txt'


if __name__ == '__main__':
    for folder_pair in folder_list:
        folder_1T, folder_2T = folder_pair
        
        if 'imb' in folder_1T:
            title_str = folder_1T.split('model_')[-1].split('_e')[0] + '_'+folder_1T.split('/')[-2] + '_val_acc1'
        else:
            title_str = folder_1T.split('model_')[-1].split('_e')[0] + '_val_acc1'
        print(title_str)
        
        txt_1T = srcRootDir + folder_1T + txtFile
        assert(os.path.exists(txt_1T))
        txt_2T = srcRootDir + folder_2T + txtFile
        assert(os.path.exists(txt_2T))
        
        epochs_1T = []
        valid_acc1_list_1T = []
        
        epochs_2T = []
        valid_acc1_list_2T = []
        
        # (1) for 1T:
        with open(txt_1T) as file:
            for line in file:
                this_line = line.rstrip()
                this_dict = json.loads(this_line)
                this_epoch = this_dict['epoch']
                this_acc = this_dict['test_acc1']
                epochs_1T.append(this_epoch)
                valid_acc1_list_1T.append(this_acc)
                #print()
                
        # (2) for 2T:
        with open(txt_2T) as file:
            for line in file:
                this_line = line.rstrip()
                this_dict = json.loads(this_line)
                this_epoch = this_dict['epoch']
                this_acc = this_dict['test_acc1']
                epochs_2T.append(this_epoch)
                valid_acc1_list_2T.append(this_acc)
                #print()
        
        #print()
        if not (len(epochs_1T) == len(epochs_2T)):
            this_len = min(len(epochs_1T),len(epochs_2T))
            epochs_1T = epochs_1T[:this_len]
            epochs_2T = epochs_2T[:this_len]
            valid_acc1_list_1T = valid_acc1_list_1T[:this_len]
            valid_acc1_list_2T = valid_acc1_list_2T[:this_len]
        assert(len(epochs_1T) == len(epochs_2T))
        
        # plot curves of val_acc1_list_orig & val_acc1_list_gan:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs_1T, valid_acc1_list_1T)
        ax.plot(epochs_2T, valid_acc1_list_2T)
        ax.legend(['valid_acc1_1T', 'valid_acc1_2SameT'])
        ax.set_ylim([0,max(max(valid_acc1_list_1T), max(valid_acc1_list_2T))+10]) #+10
        
        plt.title(title_str)
        fig.savefig(srcRootDir + 'learning_curves/'+ title_str + '.png')




