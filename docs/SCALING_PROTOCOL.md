# 1.5B ↔ 7B Scaling 对照协议（论文用）

本文档固定 **跨尺度可比** 的最小条件：同一组 task pair、同一套训练/LoRA 超参、同一 `forgetting` 定义、同一批前向谱指标（含多主方向）。**凡偏离下列任一条，不得与本文「scaling」表格直接并置。**

---

## 1. 固定条件（必须一致）

| 项 | 约定 |
|----|------|
| Task pairs | 8 个，与 `real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml` 中 `tasks` 列表**完全一致**（顺序可不同，但名称与字段须一致）。 |
| Seeds | **42, 43, 44**（与 7B smoke 主表一致；若扩样可附录补 45+，主文表须注明）。 |
| `forgetting` | `vanilla_lora` 策略下：`a_loss_after_b - a_loss_after_a`（见 `RESULTS_SUMMARY.md` §3.1）。 |
| 策略集合 | `vanilla_lora`, `random_freeze_30`, `lower_layers_freeze_30`, `c_couple_freeze_30`, `spectrum_freeze_30`（与 7B smoke yaml 一致）。 |

---

## 2. 训练/LoRA 协议（与 7B smoke 对齐）

配置文件：

- **7B（AutoDL）**：`smoke/configs/real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml`
- **1.5B（本机或任意 GPU）**：`smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol.yaml`

对齐字段（与 7B yaml 逐项相同）：

- `train.max_seq_len`, `epochs`, `batch_size`, `grad_accum_steps`, `max_train_samples`, `max_eval_samples`, `learning_rate`, `max_grad_norm`
- `lora.rank`, `lora.alpha`, `lora.target_modules`
- `metrics.c_couple_weights` 与主辅开关

**披露要求**：若因显存/时间仅能在某一尺度上微调 `max_train_samples` 等，须在论文 **limitations** 单列一行说明，且该尺度结果不得标为「主 scaling 表」。

---

## 3. 数据与离线

- **7B**：`data.local_files_only: true`，`data.hf_datasets_cache` 指向已同步的 HF datasets 根目录；模型 `model.local_path` 指向本地 7B 权重。
- **1.5B**：`data.local_files_only: true`；Windows 下将 `model.local_path` 改为本机 HuggingFace snapshot 路径；必要时设置 `data.hf_datasets_cache` 与 7B 逻辑相同。

---

## 4. 谱指标两套定义（正文须并列写清）

### 4.1 `activation_spectrum_overlap`（指标 A，rank-1，可正可负）

最后一层 hidden、batch 维句向量矩阵上，各自 `TruncatedSVD(n_components=1)` 的首主方向 **cosine**（见 `run_smoke.activation_overlap_real`）。

### 4.2 `activation_principal_cos_k3` / `activation_principal_cos_k5`（指标 B，子空间主夹角）

对 A、B 矩阵分别取前 `k_eff` 个右奇异向量张成子空间，对 `Va.T @ Vb` 做 SVD，**奇异值 = 主夹角余弦**，再取 **均值**；值域 **[0, 1]**。实现见 `run_smoke.principal_angle_mean_cos_overlap` / `activation_overlap_multi_k_real`。

**重要**：B **不是** A 的简单「加维度」；B 非负且有界，与 A 的 Pearson **不可与历史仅报告 A 的 r 直接比大小**，除非同表并列且解释量纲。

---

## 5. 跑数命令（模板）

### 5.1 多 seed 主表 + 聚合

```bash
# 仓库根目录；PYTHON 换成本机/conda/python3
PYTHON=python
CFG=smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol.yaml
OUT=$(grep ^output_dir: "$CFG" | awk '{print $2}')   # 或手写 outputs/...

$PYTHON smoke/run_multiseed.py --config "$CFG" --mode real --seeds 42 43 44 \
  --python-bin "$PYTHON" --run-smoke-path smoke/run_smoke.py
```

### 5.2 Hold-out 相关（LOSO / LOPO）

在得到 `multiseed_pair_metrics.csv` 后：

```bash
$PYTHON smoke/run_holdout_corr.py \
  --csv "$OUT/multiseed_pair_metrics.csv" \
  --target forgetting \
  --predictors activation_spectrum_overlap activation_principal_cos_k3 activation_principal_cos_k5 gradient_alignment \
  --output-csv "$OUT/holdout_corr_table.csv" \
  --output-json "$OUT/holdout_corr.json"
```

**解读约束**：LOPO 测试集仅 3 点，**探索性**；主文优先报告 **pooled** 与 **LOSO**。

---

## 6. 成功标准（建议写进附录）

- **主表**：两尺度下同一组列的 `pearson_all_seed_pair`（来自 `multiseed_summary.json`）并列。
- **稳健性**：LOSO 下 `activation_principal_cos_k3`（或 k5）的 `r_test` 符号与 pooled 是否一致（记录数值，不强求显著性检验）。
- **不宣称**：仅凭单尺度 k 指标升高即「证明因果」；scaling **闭合**需 A/B 两指标在双尺度上**同向、同序**或事先声明的可接受偏差。

---

## 7. 7B 已完成产物（参考）

- 配置：`smoke/configs/real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml`
- 输出目录：`outputs/real_smoke_qwen7b_8pairs_smoke/`（含 `multiseed_pair_metrics.csv`、`holdout_corr.json`）

1.5B 在 **同 yaml 协议** 下重跑后，将 `multiseed_summary.json` 中 `pearson_all_seed_pair` 与 7B 同名字段填入对比表即可。

---

## 7.1 当前同协议观察（2026-04）

基于已完成的 3-seed 同协议结果：

- 7B：`activation_principal_cos_k3/k5` 对 forgetting 的 pooled 相关高于 rank-1；
- 1.5B：rank-1 指标更强，`k3` 并未复现“固定最优”。

因此当前口径应为：

- ✅ “子空间覆盖相关指标有效”；
- ❌ “固定 k=3 跨尺度最优”。

下一步应优先补 `k=2` + Spearman + 反向 pair 小样本，而不是直接拟合统一 `k_opt` 公式。

---

## 8. 配置文件一览

| 环境 | 1.5B（与 7B smoke 对齐） | 7B（AutoDL） |
|------|---------------------------|--------------|
| Windows / 本机 | `smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol.yaml`（修改 `model.local_path`） | — |
| AutoDL | `smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol_autodl_local.yaml`（确认 `local_path` 存在） | `smoke/configs/real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml` |

**Windows 本机跑数**：在仓库根目录执行 `run_multiseed.py` 时建议将 stdout 重定向到文件（训练阶段可能长时间无新行），例如：

`python smoke/run_multiseed.py ... 2>&1 | Tee-Object outputs/real_smoke_qwen15b_8pairs_7b_protocol_multiseed_run.log`

跑完后在同一目录执行 §5.2 的 `run_holdout_corr.py`。
