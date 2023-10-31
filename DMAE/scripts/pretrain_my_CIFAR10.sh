#!/bin/bash
#    --imb_factor 100
#    --input_size 32

EXP_NAME=distill_base_model
GPUS=4
SAVE_DIR1="./work_dirs/${EXP_NAME}_CIFAR10_e3/imb10/"
MODEL_NAME='latest.pth'
IMAGENET_DIR='/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-10-batches-py/clean_img'
     
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --use_env main_distill.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 128 \
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
