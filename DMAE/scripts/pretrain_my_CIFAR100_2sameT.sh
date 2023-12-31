#!/bin/bash
EXP_NAME=distill_base_model
GPUS=4
SAVE_DIR1="./work_dirs/${EXP_NAME}_CIFAR100_e4s1_2sameT/imb10/"
MODEL_NAME='latest.pth'
IMAGENET_DIR='/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img'
     
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --use_env main_distill_2sameT.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 64 \
    --accum_iter 4 \
    --model mae_vit_base_patch16_dec512d8b \
    --model_teacher mae_vit_large_patch16_dec512d8b \
    --mask_ratio 0.1 \
    --epochs 200 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --teacher_model_path 'teacher_model/mae_visualize_vit_large.pth' \
    --student_reconstruction_target 'original_img' \
    --aligned_blks_indices 8 \
    --teacher_aligned_blks_indices 17 \
    --embedding_distillation_func L1 \
    --aligned_feature_projection_dim 768 1024
