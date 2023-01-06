#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t1_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set 't1_train' --test_set 'all_task_test' --num_classes 81 \
    --epochs 50 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}

##T2
EXP_DIR=exps/OWOD_t2_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --train_set 't2_train' --test_set 'all_task_test' --num_classes 81 \
    --epochs 56 --lr 2e-5 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t1_old_split/checkpoint0049.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t2_ft_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --train_set 't2_ft' --test_set 'all_task_test' --num_classes 81 \
    --epochs 101 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t2_old_split/checkpoint0054.pth' \
    ${PY_ARGS}

##T3
EXP_DIR=exps/OWOD_t3_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_train' --test_set 'all_task_test' --num_classes 81 \
    --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain "exps/OWOD_t2_ft_old_split/checkpoint0099.pth" \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t3_ft_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_ft' --test_set 'all_task_test' --num_classes 81 \
    --epochs 136 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_old_split/checkpoint0104.pth' \
    ${PY_ARGS}

##T4
EXP_DIR=exps/OWOD_t4_old_split
PY_ARGS=${@:1}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_train' --test_set 'all_task_test' --num_classes 81 \
    --epochs 141 --lr 2e-5 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_old_split/checkpoint0134.pth' \
    ${PY_ARGS}


EXP_DIR=exps/OWOD_t4_ft_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_ft' --test_set 'all_task_test' --num_classes 81 \
    --epochs 161 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100\
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t4_old_split/checkpoint0139.pth' \
    ${PY_ARGS}
    