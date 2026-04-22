# 阶段结果总结（LoRA 跨任务干涉：谱指标方向）

## 1. 这份文档回答什么问题

本阶段的目标是判断两件事：

1. 是否能找到一个对“遗忘（forgetting）”有稳定预测能力的指标。
2. 在该指标指导下，冻结策略是否已经稳定优于 vanilla LoRA。

当前结论：

- 第 1 点有正面结果（可复现）。
- 第 2 点暂未形成稳定结论（仅有轻微改善）。

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

---

## 3. 指标定义（本阶段关心）

### 3.1 `forgetting`（被预测目标）

- 定义：在 A->B 顺序微调中，B 训练后 A 任务损失相对 A 阶段后的增量。
- 公式：`forgetting = a_loss_after_b - a_loss_after_a`
- 含义：数值越大，表示 B 训练后对 A 的破坏越明显。

### 3.2 `activation_spectrum_overlap`（主指标：做了什么）

做法（每个 `(seed,pair)` 一次）：

1. 取任务 A、B 的文本样本，过模型得到最后一层 hidden states。
2. 对 token 维做均值池化，得到句向量矩阵（A 一份、B 一份）。
3. 分别对 A、B 矩阵做 `TruncatedSVD(n_components=1)`，取第一主方向向量。
4. 计算两个主方向的 cosine，得到一个标量：`activation_spectrum_overlap`。

解释：

- 该值越大，表示 A/B 在“主表征方向”上越重合；经验上更容易发生干涉与遗忘。
- 在本项目里，它用于**主预测变量**，直接与 `forgetting` 做相关分析。

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
2. **协议敏感性**：换表征层 / SVCCA 有效维数 / 激活采样条数（证明排序不是调参侥幸）。
3. **写作资产**：主图（48 点）+ 分层附图（§4.3）+ 策略与配对检验附录；策略块保持「负结果/边界」定位。

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

## 11. 论文价值判断（批判性）

**结论先说**：以当前证据量，**足够支撑一篇结构清楚、贡献边界诚实的论文**（偏 **PEFT/分析向 workshop 或 arXiv 技术报告**；冲 **顶会主会长文** 仍建议补齐下文「发表门槛」）。

**已经较强的卖点（可写进 contribution）**

1. **主指标**：`activation_spectrum_overlap` 在 48 点上对 `forgetting` 相关强（r≈0.78），且 **6/6 seed 同号**，并辅以 mixed-effects。  
2. **关键对照实验**：同一协议下 **SVCCA/CKA 标量对遗忘几乎无线性预测力**，与主指标形成尖锐对比——这是目前叙事里**最有辨识度的单组实验**，但必须配合 **§4.3 的协议说明** 使用，避免过度全称化。  
3. **诚实的负结果**：策略（尤其 `spectrum_freeze_30`）**不构成可靠收益**，且配对检验对样本构成敏感——这在方法论上是加分项（避免“硬吹策略”）。

**仍薄弱的审稿点（不补容易被追问）**

- **规模**：1.5B + 8 task pairs + 48 点，对“普适定律”类声称偏弱；**7B（或更大）+ 至少同等统计口径**是最高性价比补强。  
- **外部效度**：SVCCA/CKA 的**实现变体敏感性**（层、维数、样本量）未系统扫过。  
- **因果层级**：目前主要是预测与对照；**机制/干预**仍是讨论里需要主动承认的缺口。

**一句话建议**

- 若能在 **7B** 上复现「主指标强、SVCCA/CKA 弱（或至少显著弱于主指标）」的主排序，**论文可投性会明显上一个台阶**；若不能，仍可作为「1.5B 上的干净现象 + 局限」发表，但预期 venue 需下调。

