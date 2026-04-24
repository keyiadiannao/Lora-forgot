# 下一步执行计划（精简版，按当前事实）

## 已完成（用于写作）

- **7B 同协议 3-seed**：`outputs/real_smoke_qwen7b_8pairs_smoke/`
- **1.5B 同协议 3-seed**：`outputs/real_smoke_qwen15b_8pairs_7b_protocol/`
- **含 `k2` 列的全量重跑**（AutoDL）：上述两目录 + `outputs/real_smoke_qwen15b_reverse8pairs_7b_protocol/`、`outputs/real_smoke_qwen15b_reverse3pairs_7b_protocol/`；`holdout_corr.*` 已用 **`k2` 预测列** 重生成。
- **两侧 holdout**：均已生成 `holdout_corr.json` / `holdout_corr_table.csv`
- **协议文档**：`docs/SCALING_PROTOCOL.md`；**结果精修**：`RESULTS_SUMMARY.md` §2.3、§14（含 k2 与刷新数值）

结论提醒：当前数据支持“子空间覆盖有效，但最优 k 非固定（模型依赖）”，不支持“k=3 普适最优”。

**环境与 SSH / AutoDL 备忘**：`docs/AUTODL_AND_SSH.md`（连接方式、目录、日志与 k2 重跑要点）。

---

## P0（立即执行）

### 1) 画主图：`k vs r` 交叉图（同协议）

目标：给出 1.5B 与 7B 在相同协议下的 **`k={1,2,3,5}`** 对 forgetting 相关对照；图上同时给 Pearson 与 Spearman。（**数据已在 AutoDL 侧就绪**；将 `multiseed_pair_metrics.csv` 拷回本机后可用 `smoke/run_kcorr_table.py` 复算。）

- 输入：
  - `outputs/real_smoke_qwen15b_8pairs_7b_protocol/multiseed_pair_metrics.csv`
  - `outputs/real_smoke_qwen7b_8pairs_smoke/multiseed_pair_metrics.csv`
- 指标列：
  - `activation_spectrum_overlap`（k=1）
  - `activation_principal_cos_k2`（需用新版 `run_smoke.py` 重跑后 CSV 才有）
  - `activation_principal_cos_k3`
  - `activation_principal_cos_k5`

### 2) 补 Spearman（排除仅 Pearson 视角）

在 pooled（all seed×pair）与 LOSO 两个层面都报告 Spearman；LOPO 保留为 exploratory（n_test=3）。

---

## P0.5（高优先但次于主图）

### 3) 反向 pair 小样本验证（方向性）

目的：判断 `A->B` 与 `B->A` 是否对称，避免“无向兼容性”过度结论。

建议最小集合（3 pairs × 3 seeds）：

- `gsm8k_vs_imdb` ↔ `imdb_vs_gsm8k`
- `hotpotqa_vs_imdb` ↔ `imdb_vs_hotpotqa`
- `mmlu_vs_winograd` ↔ `winograd_vs_mmlu`

要求：除交换 A/B 外，其余超参不变。

进度：已完成反向 `3-pair×3-seed` 与 `8-pair×6-seed`；方向不对称性已确认。后续从“是否存在”转向“为何存在”（机制拆分）。

---

## P1（条件允许再做）

### 4) ~~补 `k=2`~~（已完成）

已在 `run_smoke.py` 写入 `activation_principal_cos_k2`，并在 AutoDL 上对同协议与反向配置完成重跑；论文主文可直接使用 **`k={1,2,3,5}`** 曲线。

### 5) 最小稳健性统计

至少补一个：

- bootstrap CI（对 pooled r）
- permutation test（对关键相关）

### 6) 机制拆分（方向性）

在同协议正反数据上做最小模型：

`forgetting ~ overlap + direction + overlap:direction + lora_l2_delta_b + (a_loss_after_a - a_loss_before_a)`

目标：区分“方向主效应”和“更新幅度/训练阶段差异”的贡献。

---

## 写作口径（避免踩坑）

- 不把 1.5B 旧 quick（6-seed）与同协议 7B 直接混成一张主对比图。
- 不写“多主方向总是更好”；改为“最优 k 具有模型/任务依赖性”。
- `SVCCA/CKA` 保持“本实现+本协议下弱”，避免全称化外推。

---

## 相关配置与脚本

- 1.5B 同协议（Windows）：`smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol.yaml`
- 1.5B 同协议（AutoDL）：`smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol_autodl_local.yaml`
- 1.5B 反向 3-pair（AutoDL）：`smoke/configs/real_smoke_qwen15b_reverse3pairs_7b_protocol_autodl_local.yaml`
- 7B 同协议（AutoDL）：`smoke/configs/real_smoke_qwen7b_8pairs_smoke_autodl_local.yaml`
- holdout 脚本：`smoke/run_holdout_corr.py`
- k-r 表脚本：`smoke/run_kcorr_table.py`（已生成：`outputs/scaling_kcorr_table.csv`）
