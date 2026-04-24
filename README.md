# LoRA 跨任务干涉烟雾测试（最小可执行版）

这个工程用于快速验证两件事：

1. 任务耦合指标（梯度 / 激活谱 / Fisher）是否能预测遗忘趋势  
2. 基于耦合指标的结构感知冻结是否优于随机冻结

## 1) 环境准备

Windows PowerShell 下执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

如果你本机已经有 conda，推荐直接使用（当前验证可用环境：`C:\Users\26433\miniconda3\envs\mamba2`）：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" -m pip install -r requirements.txt
```

## 2) 快速验证链路（不训练）

```powershell
py smoke\run_smoke.py --config smoke\configs\local_smoke.yaml --dry-run
```

或使用 conda 指定解释器：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\local_smoke.yaml --dry-run
```

这一步会打印任务配置、指标配置和策略配置，确保流程正确。

## 3) 正式跑本地烟雾测试

```powershell
py smoke\run_smoke.py --config smoke\configs\local_smoke.yaml
```

运行后会在 `outputs\` 目录生成：

- `pair_metrics.csv`: 每个任务对的单指标与遗忘值
- `summary.json`: 汇总相关性与策略对比
- `pareto_points.csv`: 保持率-新任务增益点

## 4) 切换到完整版

把配置改成：

```powershell
py smoke\run_smoke.py --config smoke\configs\full_smoke.yaml
```

## 4.1) 真实顺序微调模式（A->B）

先用 tiny 配置验证完整链路（真实训练、真实遗忘）：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\real_smoke_tiny.yaml --mode real
```

结果同样输出到 `outputs/`：

- `pair_metrics.csv`: 指标与 vanilla 遗忘
- `pareto_points.csv`: 各策略保持率与新任务增益
- `summary.json`: 相关性汇总

## 4.2) 复用本地已有模型权重（推荐）

如果你已在本地缓存过模型，优先使用本地离线模式避免重复下载：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\real_smoke_gpt2_local.yaml --mode real
```

该配置使用：

- `model.name: gpt2`
- `model.local_files_only: true`

## 4.3) Qwen1.5B 本地离线烟雾测试

如果你已补全 `Qwen/Qwen2.5-1.5B-Instruct` 本地权重，可直接运行：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\real_smoke_qwen15b_local.yaml --mode real
```

## 4.4) Qwen1.5B 四任务对 Full（单 seed）

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\real_smoke_qwen15b_full.yaml --mode real
```

## 4.5) 多 seed 批跑 + 自动汇总（推荐判定 H1/H2）

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_multiseed.py --config smoke\configs\real_smoke_qwen15b_full.yaml --mode real --seeds 42 43 44 --python-bin "C:\Users\26433\miniconda3\envs\mamba2\python.exe"
```

批跑输出：

- `multiseed_summary.json`: H1/H2 判定与统计
- `multiseed_corr.csv`: 各 seed 的相关性
- `multiseed_pair_metrics.csv`: 合并后的任务对指标
- `multiseed_pareto.csv`: 合并后的策略对比明细

## 4.6) 强化遗忘诊断配置（先确认会遗忘）

先跑强度版（2 任务对），验证是否出现明显 forgetting 和参数更新：

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_smoke.py --config smoke\configs\real_smoke_qwen15b_local_strong.yaml --mode real
```

输出的 `pareto_points.csv` 里会包含诊断字段：

- `a_loss_before_a`, `a_loss_after_a`, `a_loss_after_b`
- `b_train_avg_loss`, `b_train_steps`
- `lora_l2_before_b`, `lora_l2_after_b`, `lora_l2_delta_b`

## 4.7) 8 任务对（提升统计功效）

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_multiseed.py --config smoke\configs\real_smoke_qwen15b_8pairs.yaml --mode real --seeds 42 43 --python-bin "C:\Users\26433\miniconda3\envs\mamba2\python.exe"
```

说明：`multiseed_summary.json` 中新增 `n_points_seed_pair`，当点数 < 15 时不会给出 H1 通过结论。

## 4.8) 1.5B↔7B 同协议 scaling 与 k–r 表

- 协议与披露约束：`docs/SCALING_PROTOCOL.md`
- 合并 `multiseed_pair_metrics.csv` 上的 Pearson/Spearman（含 **`k2`**）：`smoke/run_kcorr_table.py`
- AutoDL 连接与日志路径备忘：`docs/AUTODL_AND_SSH.md`

## 5) 注意事项

- 首次下载模型和数据集会较慢
- 8GB 显存建议先用 `Qwen/Qwen2.5-0.5B-Instruct` 或同级别 0.5B~1.5B
- 显存不够时优先降 `max_seq_len`、`batch_size`、`lora_rank`
