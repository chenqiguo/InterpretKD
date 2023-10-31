#!/bin/bash
EXP_NAME=distill_finetuned_base_model
GPUS=4
SAVE_DIR1="./work_dirs/${EXP_NAME}_CIFAR100_e4s1_2sameT/imb10/"

IMAGENET_DIR='/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img'

FINETUNE_EXP_FOLDER='distill_base_model_CIFAR100_e4s1_2sameT/imb10'
FINETUNE_MODEL_NAME='checkpoint-199.pth'


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_finetune.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune "./work_dirs/${FINETUNE_EXP_FOLDER}/${FINETUNE_MODEL_NAME}" \
    --epochs 1000 \
    --blr 5e-4 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --data_path ${IMAGENET_DIR} \
    --seed 0 --nb_classes 100
