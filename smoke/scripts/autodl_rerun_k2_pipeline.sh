#!/usr/bin/env bash
# AutoDL：为写入 activation_principal_cos_k2 等列，按既有 yaml 顺序重跑 real multiseed（串行，单卡安全）。
set -euo pipefail
cd /root/work/Lora-forgot
export HF_HOME=/root/autodl-tmp/hf/home
export HF_HUB_CACHE=/root/autodl-tmp/hf/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers
export TORCH_HOME=/root/autodl-tmp/torch
PY=/root/miniconda3/bin/python
RS=smoke/run_smoke.py

log() { echo "[$(date -Is)] $*"; }

log "start k2 full rerun (4 configs, sequential)"
$PY -u smoke/run_multiseed.py \
  --config smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol_autodl_local.yaml \
  --mode real --seeds 42 43 44 --python-bin "$PY" --run-smoke-path "$RS" --no-probe
log "done 15b 8pairs 7b_protocol"

$PY -u smoke/run_multiseed.py \
  --config smoke/configs/real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml \
  --mode real --seeds 42 43 44 --python-bin "$PY" --run-smoke-path "$RS" --no-probe
log "done 7b 8pairs"

$PY -u smoke/run_multiseed.py \
  --config smoke/configs/real_smoke_qwen15b_reverse8pairs_7b_protocol_autodl_local.yaml \
  --mode real --seeds 42 43 44 45 46 47 --python-bin "$PY" --run-smoke-path "$RS" --no-probe
log "done 15b reverse8"

$PY -u smoke/run_multiseed.py \
  --config smoke/configs/real_smoke_qwen15b_reverse3pairs_7b_protocol_autodl_local.yaml \
  --mode real --seeds 42 43 44 --python-bin "$PY" --run-smoke-path "$RS" --no-probe
log "done 15b reverse3"

log "ALL DONE"
