# 阶段结果总结（LoRA 跨任务干涉：谱指标方向）

## 1. 这份文档回答什么问题

本阶段的目标是判断两件事：

1. 是否能找到一个对“遗忘（forgetting）”有稳定预测能力的指标。
2. 在该指标指导下，冻结策略是否已经稳定优于 vanilla LoRA。

当前结论：

- 第 1 点有正面结果（可复现）。
- 第 2 点暂未形成稳定结论（仅有轻微改善）。

**1.5B↔7B 同协议 scaling 表**：指标 A/B 定义、yaml、命令与 hold-out 解读约束见 **`docs/SCALING_PROTOCOL.md`**；1.5B 对齐配置为 `smoke/configs/real_smoke_qwen15b_8pairs_7b_protocol.yaml`。

---

## 2. 数据与实验设置（本阶段）

### 2.1 主验证集（推荐作为主结果）

路径：`outputs/real_smoke_qwen15b_8pairs_quick`

- 模型：`Qwen2.5-1.5B-Instruct`（本地快照离线）
- 任务对：8 pairs
- seed：`42, 43, 44, 45, 46, 47`（共 6 seeds）
- 总点数：`48`（6 seeds × 8 pairs）
- 8 个 pair 明细（每个 pair 都有 6 个 seed 样本）：
  - `gsm8k_vs_2wiki`
  - `gsm8k_vs_hotpotqa`
  - `gsm8k_vs_imdb`
  - `gsm8k_vs_winograd`
  - `hotpotqa_vs_imdb`
  - `mmlu_vs_imdb`
  - `mmlu_vs_winograd`
  - `twowiki_vs_winograd`

关键输出：

- `multiseed_summary.json`
- `probe_report.json`
- `focus_report.json`
- `p0_stats_report.json`（P0 统计检验：mixed-effects + 配对检验）

### 2.2 小规模探测集（只作补充，不作主结论）

路径：`outputs/probe_smoke_qwen15b_3pairs`

- 任务对：3 pairs
- seed：`46-51`
- 总点数：`18`

该集合相关值更高，但样本/覆盖较小，不建议单独当主结果。

### 2.3 同协议 scaling 对照（新增，当前最关键）

为避免“协议不一致”的比较偏差，我们补跑了与 7B smoke 对齐的 1.5B 协议（同 8 pairs、同 seeds 42/43/44、同 train/lora 超参），并与 7B 三 seed 结果并列。

- 7B 路径：`outputs/real_smoke_qwen7b_8pairs_smoke`
- 1.5B 路径：`outputs/real_smoke_qwen15b_8pairs_7b_protocol`
- hold-out 路径：两侧均有 `holdout_corr.json`

当前 pooled Pearson（与 `multiseed_pair_metrics.csv` 全 24 点一致；来自各输出目录 **`holdout_corr.json`** 的 `pearson_pooled_all_rows`；2026-04 含 **`k2` 列** 全量重跑后刷新）：

| 模型 | k1 (`activation_spectrum_overlap`) | k2 (`activation_principal_cos_k2`) | k3 (`activation_principal_cos_k3`) | k5 (`activation_principal_cos_k5`) | `gradient_alignment` |
|------|------------------------------------:|-------------------------------------:|-------------------------------------:|-------------------------------------:|----------------------:|
| 1.5B（同协议 3 seeds） | **0.869** | -0.020 | 0.045 | 0.460 | -0.086 |
| 7B（同协议 3 seeds） | 0.433 | 0.807 | **0.865** | **0.881** | 0.169 |

同批数据的 pooled Spearman（`smoke/run_kcorr_table.py` 对同一 CSV 直接计算）：

| 模型 | k1 | k2 | k3 | k5 | grad |
|------|---:|---:|---:|---:|-----:|
| 1.5B | **0.832** | -0.141 | -0.065 | 0.166 | -0.043 |
| 7B | 0.363 | 0.691 | **0.823** | **0.845** | 0.126 |

解读（必须保守）：

- **7B**：在 Pearson 上 **`k1 < k2 < k3 < k5` 单调上升**，`k2` 已明显高于 rank-1；多主方向秩在本协议下整体优于只看第一层主方向。
- **1.5B 同协议**：**rank-1（k1）仍最强**；`k2/k3` 接近零或弱负（Spearman 上 `k2` 亦为负），`k5` 为中等强度正相关——不能写成“秩越高越好”。
- 因此“固定 k=3”不成立；更合理命题是：**`k_opt` 具有模型/任务依赖性**；补 **`k=2`** 后叙事从“只在 1/3/5 插值”升级为“**7B 单调、1.5B 非单调**”的可检验陈述。

---

## 3. 指标定义（本阶段关心）

### 3.1 `forgetting`（被预测目标）

- 定义：在 A->B 顺序微调中，B 训练后 A 任务损失相对 A 阶段后的增量。
- 公式：`forgetting = a_loss_after_b - a_loss_after_a`
- 含义：数值越大，表示 B 训练后对 A 的破坏越明显。

### 3.2 `activation_spectrum_overlap`（指标 A：rank-1 谱重叠）

做法（每个 `(seed,pair)` 一次）：

1. 取任务 A、B 的文本样本，过模型得到最后一层 hidden states。
2. 对 token 维做均值池化，得到句向量矩阵（A 一份、B 一份）。
3. 分别对 A、B 矩阵做 `TruncatedSVD(n_components=1)`，取第一主方向向量。
4. 计算两个主方向的 cosine，得到一个标量：`activation_spectrum_overlap`。

解释：

- 该值越大，表示 A/B 在“主表征方向”上越重合；经验上更容易发生干涉与遗忘。
- 在本项目里，它用于**主预测变量**，直接与 `forgetting` 做相关分析。

### 3.2b `activation_principal_cos_k2` / `k3` / `k5`（指标 B：多主方向子空间）

- **目的**：检验“单主方向是否在更大模型中失真”，用于 **1.5B↔7B scaling** 对照（与指标 A **并列报告**，不可混为同一数值序列）。
- **做法**：对 A、B 的句向量矩阵分别 `TruncatedSVD(n_components=k_eff)`，`k_eff ∈ {2,3,5}`，取各自前 `k_eff` 个右奇异向量张成子空间；对 `Va.T @ Vb` 做 SVD，**奇异值 = 两子空间主夹角余弦**，再取 **均值**；结果在 **[0, 1]**。
- **实现**：`run_smoke.py` 中 `principal_angle_mean_cos_overlap` / `activation_overlap_multi_k_real`；`pair_metrics.csv` 列名 `activation_principal_cos_k2`（及 k3、k5）。
- **协议与跑数命令**：见仓库根目录 **`docs/SCALING_PROTOCOL.md`**（含与 7B smoke 对齐的 yaml 名、多 seed、`run_holdout_corr.py` 用法）。

### 3.3 `spectrum_layers_std/span`（辅指标：做了什么）

先按层算一组谱重叠分数，再做层间统计：

1. 对每个可识别 decoder layer，重复“取 A/B 该层表示 -> 主方向 -> cosine”的过程，得到该层分数 `s_l`。
2. 形成层分数集合 `{s_l}` 后，计算：
   - `spectrum_layers_std = std({s_l})`
   - `spectrum_layers_span = max({s_l}) - min({s_l})`

解释：

- 这两个量不直接预测策略收益，而是描述“跨层谱异质性”。
- 在本项目里，它们作为**辅助解释变量**，用于回答：
  - 为什么有些 pair 遗忘重、有些轻；
  - 仅看最后一层标量是否遗漏了跨层结构信息。

### 3.4 主+辅在分析中的分工

- 主：`activation_spectrum_overlap`（看是否稳定预测 forgetting）。
- 辅：`spectrum_layers_std/span`（看层间结构是否提供额外解释）。
- 当前证据：主指标最强；辅指标提供补充；与同一表征口径下的 SVCCA/CKA 标量对照见 **§4.3**。

---

## 4. 主结果（8pairs, 6 seeds, 48 点）

数据来源：`outputs/real_smoke_qwen15b_8pairs_quick/multiseed_summary.json`

### 4.1 相关性（all seed×pair）

（本轮已补全 6 seed 输出，并在同一套前向表征上增加 `svcca_overlap` / `linear_cka_overlap` 基线。）

- `activation_spectrum_overlap vs forgetting`：**r = 0.775**
- `spectrum_layers_std vs forgetting`：**r = -0.395**
- `spectrum_layers_span vs forgetting`：**r = -0.301**
- `svcca_overlap vs forgetting`：**r = 0.119**
- `linear_cka_overlap vs forgetting`：**r = 0.023**

对照解读（48 点、同最后一层 hidden 表征口径）：

- 单主方向谱重叠（`activation_spectrum_overlap`）仍显著强于「线性 CKA」与「简化 SVCCA」标量；后两者在本设定下对 forgetting 几乎不构成有效预测。

### 4.2 跨 seed 稳定性（来自 focus_report）

- `activation_spectrum_overlap` 的 seed 内相关同号一致率：**1.0（6/6）**
- `activation_spectrum_overlap` 的 seed 内相关均值/波动：`mean=0.786, std=0.053`

结论：

- `activation_spectrum_overlap` 对 forgetting 的预测信号是稳定且可复现的。
- `spectrum_layers_std` 仍有中等强度补充；`span` 在本轮全量 48 点上相关减弱，更适合与分层子集（正/负遗忘）对照解读，而不是单独抬成与 `std` 对称的“主辅结论”。

### 4.3 SVCCA/CKA 对照与分层（探索性）

数据来源：`outputs/real_smoke_qwen15b_8pairs_quick/focus_report.json`（与 4.1 同一 48 个 `(seed,pair)` 点）。

**协议（务必在论文里写清，避免过度外推）**

- 与 `activation_spectrum_overlap` **共享同一表征口径**：最后一层 hidden、token-mean 池化、与主实验一致的前向采样与样本量上限。
- `svcca_overlap`、`linear_cka_overlap` 为本仓库实现的**轻量标量基线**，用于与主指标公平对照；**不应等同于**文献中所有可能的 SVCCA/CKA 变体（层集合、成分数、白化、对齐方式等均可改变数值）。

**全量 48 点（复述自 4.1）**

- `svcca_overlap vs forgetting`：r = **0.119**；`linear_cka_overlap vs forgetting`：r = **0.023**（相对主指标 r≈0.775 可视为「几乎无线性预测力」）。

**分层（探索性；子集样本量有限）**

- 定义：`forgetting > 0` 为正遗忘子集 **n = 25**；`forgetting ≤ 0` 为非正遗忘子集 **n = 23**（与 `probe_report` 分层一致）。
- 与 `forgetting` 的 Pearson（子集内）：

| 指标 | 正遗忘子集 (n=25) | 非正遗忘子集 (n=23) |
|------|-------------------|---------------------|
| `activation_spectrum_overlap` | **+0.325** | **+0.554** |
| `svcca_overlap` | **-0.776** | **-0.539** |
| `linear_cka_overlap` | **-0.821** | **+0.629** |

**可写进讨论的一句话框架（保守版）**

- 全样本上 SVCCA/CKA 标量接近 0 的线性相关，**可能部分来自子集间方向/强度不一致**（尤其 `linear_cka_overlap` 在两子集符号相反）；这与「单一全空间相似度标量难以充当遗忘的充分统计量」相容。
- **不建议**据此在正文里推出强因果；更适合标注为 **exploratory**，并配合 7B 与协议敏感性再检验。

---

## 5. 策略结果（为什么暂不作为主贡献）

数据来源：`outputs/real_smoke_qwen15b_8pairs_quick/focus_report.json`

相对 vanilla 的全体平均变化（48 点）：

- `random_freeze_30`：`delta_forgetting ≈ -0.0327`, `delta_gain ≈ +0.00026`
- `lower_layers_freeze_30`：`delta_forgetting ≈ -0.0229`, `delta_gain ≈ +0.00019`
- `c_couple_freeze_30`：`delta_forgetting ≈ -0.0218`, `delta_gain ≈ -0.00052`
- `spectrum_freeze_30`：`delta_forgetting ≈ -0.0155`, `delta_gain ≈ -0.00253`

解释：

- 平均上有小幅改善，但幅度小、非劣率不高，不足以构成“策略稳定优于 baseline”的主结论。
- 显著性检验（`p0_stats_report.json`，本轮 48 点重聚合后）：`spectrum_freeze_30` 相对 `random_freeze_30` 在 `forgetting` 上 **略差且接近显著**（`delta_forgetting` 均值约 `+0.017`，单侧“更差”含义下 `t-test p≈0.050`；Wilcoxon `p≈0.068`）；`new_task_gain` 上仍不显著（`t-test p≈0.13`）。

---

## 6. 本阶段可对外表达的结论（建议口径）

推荐口径：

1. 在 8pairs × 6 seeds（48 点）上，`activation_spectrum_overlap` 与遗忘呈稳定强相关（r≈0.78，跨 seed 同号）。
2. 同口径下 `svcca_overlap` / `linear_cka_overlap` 对遗忘几乎无预测力（|r|≪主指标），支持「**在本协议下**，单主方向谱指标远强于这两个子空间类标量」的对照叙事（见 §4.3 协议说明，避免全称化到所有 SVCCA/CKA 变体）。
3. 按层谱统计：`std` 仍有中等补充（|r|≈0.40）；`span` 在全量 48 点上相关减弱（|r|≈0.30），更适合结合子集与机制实验解读。
4. 冻结策略在均值上有轻微收益，但稳定性不足，暂不作为主结果。
5. 同协议 1.5B↔7B 对照显示：`k_opt` 非固定常数；“k=3 普适最优”不成立，需改为“子空间维度最优值具有模型依赖性”。

不建议口径：

- “策略已显著优于 vanilla”
- “已证明原 H1/H2 全部成立”

---

## 7. 当前风险与边界

- 相关不等于因果，仍需机制验证（如层干预/表示对齐检验）。
- 指标信号强，但策略收益尚不稳，说明“可预测”不直接等于“可控制”。
- 小规模 probe（18 点）可以支持趋势，但主结论应以 48 点结果为准。
- SVCCA/CKA 为**实现依赖的基线对照**：正文应明确协议；避免将「本实现下的弱预测」泛化为「一切子空间方法失效」。

---

## 8. 下一步（建议）

更细的实验排期与命令模板见仓库根目录 **`NEXT_STEPS.md`**（已按「论文可投性」更新）。此处仅保留方向：

1. **外部效度**：至少一套 **7B（或当前环境能稳定跑的最大档）** 复刻主表（主指标 + SVCCA/CKA + mixed-effects）。
2. **k-曲线与秩敏感性**：补 `k={1,2,3,5}` 的 `k vs r` 交叉图（Pearson+Spearman），至少先补 `k=2`。
3. **方向性检查**：代表性 2-3 个 pair 做反向 `B->A`，验证“无向相似度 vs 有向遗忘”的叙事边界。
4. **写作资产**：主图（同协议 1.5B/7B）+ 分层附图（§4.3）+ 策略与配对检验附录；策略块保持「负结果/边界」定位。

---

## 9. P0 统计检验（新增）

数据来源：`outputs/real_smoke_qwen15b_8pairs_quick/p0_stats_report.json`

### 9.1 mixed-effects（目标：`forgetting`）

- 模型1：`forgetting ~ activation_spectrum_overlap + (1|seed) + (1|pair)`  
  - `activation_spectrum_overlap` 系数为正（约 `+3.52`），方向与相关分析一致。  
  - 拟合带 Hessian 正定性警告（方差分量在边界附近常见）。
- 模型2：`forgetting ~ activation_spectrum_overlap + spectrum_layers_std + spectrum_layers_span + (1|seed) + (1|pair)`  
  - `activation_spectrum_overlap` 仍显著为正（约 `+3.32`, `p` 极小）。  
  - `spectrum_layers_std`、`spectrum_layers_span` 在该联合模型下 **均未达到常规显著**（`p≈0.44` / `p≈0.17`）。  
  - 该次拟合未再报 Hessian 警告（以 `p0_stats_report.json` 为准）。

### 9.2 配对检验：`spectrum_freeze_30` vs `random_freeze_30`（48 个 seed×pair 配对点）

- `delta_forgetting = spectrum - random`（负值更好）  
  - 均值约 `+0.017`，`t-test p≈0.050`，`Wilcoxon p≈0.068`
- `delta_gain = spectrum - random`（正值更好）  
  - 均值约 `-0.00278`，`t-test p≈0.134`，`Wilcoxon p≈0.061`

结论：在本轮 48 点数据上，`spectrum_freeze_30` 相对 `random_freeze_30` **没有表现出可靠收益**；在 `forgetting` 上甚至呈现“略差且统计上接近显著”的迹象（解释需谨慎，但足以否定“谱冻结已优于随机冻结”的说法）。

---

## 10. 复现实验命令（本阶段关键）

```bash
# 8pairs 6-seed：若仅需重聚合已有 seed_* 输出
python smoke/run_multiseed.py --config smoke/configs/real_smoke_qwen15b_8pairs_quick.yaml --seeds 42 43 44 45 46 47 --aggregate-only --no-probe
# 若只补跑单个 seed（示例：47）再聚合
python smoke/run_smoke.py --config smoke/configs/real_smoke_qwen15b_8pairs_quick.yaml --mode real --seed 47 --output-dir outputs/real_smoke_qwen15b_8pairs_quick/seed_47
python smoke/run_multiseed.py --config smoke/configs/real_smoke_qwen15b_8pairs_quick.yaml --seeds 42 43 44 45 46 47 --aggregate-only --no-probe
python smoke/run_probe_report.py --input-dir outputs/real_smoke_qwen15b_8pairs_quick
python smoke/run_focus_report.py --input-dir outputs/real_smoke_qwen15b_8pairs_quick
python smoke/run_p0_stats.py --input-dir outputs/real_smoke_qwen15b_8pairs_quick
```

```bash
# 3pairs 6-seed（补充验证）
python smoke/run_multiseed.py --config smoke/configs/probe_smoke_qwen15b_3pairs.yaml --mode real --seeds 46 47 48 49 50 51 --aggregate-only --no-probe
python smoke/run_probe_report.py --input-dir outputs/probe_smoke_qwen15b_3pairs
python smoke/run_focus_report.py --input-dir outputs/probe_smoke_qwen15b_3pairs
```

---

## 11. 论文价值判断（更新版，按当前证据）

**结论先说**：以当前证据量，已经可支持一篇“现象扎实、边界诚实”的论文（PEFT/分析向 workshop 或 arXiv 技术报告）；若要冲更强 venue，仍需补方向性与秩敏感性。

**当前可主张的贡献（建议写法）**

1. **同协议跨尺度结果已形成**：1.5B 与 7B 均完成 8-pair×3-seed，对照口径统一（见 `docs/SCALING_PROTOCOL.md`）。
2. **低维覆盖信号成立，但最优秩不固定**：7B 同协议上 **Pearson 满足 `k1<k2<k3<k5`**；1.5B 同协议上 **k1 仍最强**，`k2/k3` 弱。应表述为“`k_opt` 模型依赖”，而非“固定 k=3 普适最优”。
3. **全局子空间基线在本协议下弱**：SVCCA/CKA 标量对 forgetting 的解释力整体弱于主谱指标（实现依赖，需保留协议限定语）。
4. **负结果完整披露**：冻结策略增益不稳定，`spectrum_freeze_30` 未表现出可靠优势（含配对检验）。

**仍需补强的审稿风险点**

- **方向性**：当前主表以 A->B 为主，需用反向小样本（B->A）明确“无向兼容性 vs 有向脆弱性”的边界。
- **秩分辨率**：已补 **`k=2`**（同协议 1.5B/7B 与反向 8-pair 重跑）；主文可给 **`k vs r`**（Pearson+Spearman）折线，并明确 7B 单调、1.5B 非单调。
- **因果层级**：现阶段仍是预测/相关，不是机制证明；正文需主动承认这一点。

**一句话建议**

- 这项工作最稳妥的定位是：**提出并验证“任务干扰由低维子空间覆盖驱动，且最优覆盖秩随模型尺度变化”的经验规律**；把“统一理论”保持为动机，而不是已证明定理。

---

## 12. 叙事框架（精炼版：保留与降温）

本节用于统一最近讨论，避免在正文中“过强外推”。

### 12.1 可保留（适合写入动机/讨论）

1. **子空间覆盖统一语言**：将 LoRA 遗忘、长度泛化、跨模态迁移统一为“源计算子空间对目标子空间的覆盖程度”。
2. **自由能/ELBO 直觉可用**：可将 LoRA 微调解释为“拟合项（新任务）与偏离项（遗忘）”的权衡；用于 intuition，不作严格证明。
3. **SVCCA/CKA 失效的工作假设**：全空间相似度可能被冗余维主导，而任务干扰更受低维有效子空间影响。

### 12.2 必须降温（当前不能直接当结论）

1. **宏大哲学命题不上主文**：例如“物质本质是信息”这类表达，容易偏离技术审稿标准。
2. **固定 `k=3` 最优不成立**：同协议结果显示 7B 与 1.5B 的最优秩并不一致。
3. **几何量≠信息论量的严格等价**：`subspace cosine`、互信息、条件熵目前是类比关系，不是已证明同构。

### 12.3 当前可执行论文叙事

- **主命题（稳）**：LoRA 遗忘更受任务相关低维子空间覆盖关系影响，而非整体表示相似度。
- **次命题（有边界）**：最优覆盖尺度具有模型依赖性（存在 scaling interaction）。
- **负结论（可成亮点）**：全局相似度标量在跨尺度上不稳定，难作统一预测器。

---

## 13. 反向 3-pair 验证（B->A，探索性）

数据来源：`outputs/real_smoke_qwen15b_reverse3pairs_7b_protocol`（3 pairs × 3 seeds = 9 点）。

任务：

- `imdb_vs_gsm8k`
- `imdb_vs_hotpotqa`
- `winograd_vs_mmlu`

pooled 相关（`multiseed_pair_metrics.csv`）：

- `activation_spectrum_overlap`：Pearson `0.353`，Spearman `0.600`
- `activation_principal_cos_k3`：Pearson `0.359`，Spearman `-0.033`
- `activation_principal_cos_k5`：Pearson `0.587`，Spearman `0.367`
- `gradient_alignment`：Pearson `-0.269`，Spearman `-0.367`

解读（仅探索性）：

- 反向方向下各指标排序与 8-pair 主表并不完全一致，提示“方向性 + 任务簇”交互存在。
- 当前样本仅 9 点，不足以给稳定统计结论；该结果用于支持“需要方向性章节”，不用于替代主表结论。

---

## 14. 反向 8-pair × 6-seed（48 点，新增主证据）

数据来源：`outputs/real_smoke_qwen15b_reverse8pairs_7b_protocol`。

### 14.1 反向方向自身（B->A，48 点）

`smoke/run_kcorr_table.py` 对 `multiseed_pair_metrics.csv`（48 点）：

| 指标 | Pearson | Spearman |
|------|--------:|---------:|
| `activation_spectrum_overlap` (k1) | 0.237 | 0.044 |
| `activation_principal_cos_k2` | -0.240 | -0.368 |
| `activation_principal_cos_k3` | -0.402 | -0.600 |
| `activation_principal_cos_k5` | 0.175 | -0.014 |
| `gradient_alignment` | -0.463 | -0.402 |

结论：反向方向上，正向里最强的 **k1** 信号减弱；**k3** 在 Pearson/Spearman 上均为负；**k2** 亦呈负相关，与 7B 正向的 **`k1<k2`** 结构形成对照。

### 14.2 正反方向对照（同协议、同 seeds 42/43/44）

为保证口径一致，使用：

- 正向：`outputs/real_smoke_qwen15b_8pairs_7b_protocol/multiseed_pair_metrics.csv`
- 反向：`outputs/real_smoke_qwen15b_reverse8pairs_7b_protocol/multiseed_pair_metrics.csv`
- 统一 seed 过滤：42/43/44（各 24 点）
- 汇总：`outputs/directional_forward_vs_reverse_seed4244.csv/json`

对照结果：

| 方向 | k1 | k2 | k3 | k5 |
|------|----|----|----|-----|
| Forward (A->B) | 0.869 / 0.832 | -0.020 / -0.141 | 0.045 / -0.065 | 0.460 / 0.166 |
| Reverse (B->A) | 0.219 / 0.002 | -0.245 / -0.388 | -0.415 / -0.651 | 0.124 / -0.017 |

（表中格式均为 **Pearson / Spearman**；子集为 seeds **42/43/44** 各 8 pair，共 24 点。）

这在同协议同 seed 子集上确认了：**方向不对称性存在**（不仅是 3-pair 小样本噪声）；**k2** 在反向为负、在正向 1.5B 亦近零/略负，与 **7B 正向 k2≫k1** 形成 scaling 对照锚点。

### 14.3 解释边界

- 当前不对称性可由多种机制共同造成：锚定表示差异、任务学习速度差异、更新幅度差异（`lora_l2_delta`）。
- 本阶段尚不能把差异唯一归因到某一种机制；因此“机制解释”仍应写作工作假设。

