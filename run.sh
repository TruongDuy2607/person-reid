#!/usr/bin/env bash
# ============================================================
#  SOLIDER-REID  --  Unified ReID Training (DDP, 4 GPUs)
#  Datasets: duke + market + iust + cuhk03 + vnpt1
# ============================================================

# ---------- Swin Base (SOLIDER pretrain, checkpoint_tea.pth) ----------
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=1 \
    train.py --config_file configs/unified/swin_base.yml

# ---------- Swin Small (ImageNet pretrain) ----------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --master_port 29502 \
#     --nproc_per_node=4 \
#     train.py --config_file configs/unified/swin_small.yml

# ---------- Swin Tiny (ImageNet pretrain) ----------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --master_port 29503 \
#     --nproc_per_node=4 \
#     train.py --config_file configs/unified/swin_tiny.yml

# ============================================================
#  Single-GPU runs (for debugging / ablation)
# ============================================================

# Swin Base  – single GPU
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/unified/swin_base.yml \
#     MODEL.DIST_TRAIN False SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.0008

# Market-1501 baseline (original, single GPU)
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/market/swin_base.yml \
#     MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'weights/checkpoint_tea.pth' \
#     OUTPUT_DIR './log/market/swin_base' SOLVER.BASE_LR 0.0002 \
#     SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2
