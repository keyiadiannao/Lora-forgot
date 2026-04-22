# 下一步执行计划（论文导向）

## 当前已收口（可写进主文/附录）

- **主指标**：`activation_spectrum_overlap`（48 点，6 seeds×8 pairs），`focus_report` 中跨 seed 同号率 1.0。
- **对照实验**：同表征口径下的 `svcca_overlap`、`linear_cka_overlap` vs `forgetting`（全量弱；分层见 `RESULTS_SUMMARY.md` §4.3）。
- **统计**：`p0_stats_report.json`（mixed-effects + `spectrum_freeze_30` vs `random_freeze_30` 配对检验）。
- **工程**：`run_compat_scan.py`（训练前风险扫描）、`run_probe_report.py` / `run_focus_report.py` / `run_p0_stats.py`。

## 发表门槛（按优先级）

### P0（最高）：7B（或环境内最大可稳跑档）复刻主表

- **目的**：堵「仅小模型」质疑；验证主排序是否保持：**主指标 ≫ SVCCA/CKA**。
- **建议最小集**：同一 8 pairs；若显存紧，可先 **3–4 seeds** 做趋势复验，再补满 6 seeds。
- **实现要点**：
  - 复制 `smoke/configs/real_smoke_qwen15b_8pairs_quick.yaml` 为新文件（例如 `smoke/configs/real_smoke_qwen7b_8pairs_quick.yaml`），修改 **`model.name` / `model.local_path`**、`**output_dir**`（勿覆盖 1.5B 目录）；必要时下调 `train.max_train_samples`、`train.max_seq_len`、`train.batch_size` 以适配显存，并在论文中**披露**与 1.5B 的差异。
- **跑完后**（与 1.5B 相同流水线；下面 `--config` / `--input-dir` 换成你的 7B 配置与输出目录）：

```powershell
Set-Location "d:\cursor_try\claude_codex"
$py = "C:\Users\26433\miniconda3\envs\mamba2\python.exe"
$cfg = "smoke\configs\real_smoke_qwen7b_8pairs_quick.yaml"
$out = "outputs\real_smoke_qwen7b_8pairs_quick"
& $py smoke\run_multiseed.py --config $cfg --mode real --seeds 42 43 44 --python-bin $py --run-smoke-path smoke\run_smoke.py --no-probe
& $py smoke\run_multiseed.py --config $cfg --seeds 42 43 44 --aggregate-only --no-probe --python-bin $py --run-smoke-path smoke\run_smoke.py
& $py smoke\run_probe_report.py --input-dir $out
& $py smoke\run_focus_report.py --input-dir $out
& $py smoke\run_p0_stats.py --input-dir $out
```

### P1：协议敏感性（短平快，支撑「不是调参侥幸」）

在同一 1.5B + 48 点上做 **ablation**（每次改一个旋钮，重跑前向指标块即可；若不想重训，可单独写 `run_metric_ablation.py` 调 `run_smoke` 里收集 hidden 的函数——当前未拆脚本则整轮 smoke 成本高）：

| 旋钮 | 建议取值 |
|------|----------|
| 表征层 | 最后一层 vs 中间层均值 vs 多层平均 |
| SVCCA 截断维数 | 例如 `max_components ∈ {10, 20, 40}` |
| 激活采样条数 | 与当前 `≥32` 截断一致 vs 加倍（看相关是否排序不变） |

成功标准：主指标排序稳定；SVCCA/CKA **至多弱于主指标**（不要求数值完全复现）。

### P2：负对照（可选但加分）

- **任务标签置换**或 **跨 pair 打乱 forgetting 对齐**：相关应崩塌，用于防「流水线泄漏」质疑。
- 可与 P1 共用轻量脚本，不必每个都跑满 SFT。

### P3：写作资产（与实验并行）

- 主图：48 点 `activation_spectrum_overlap` vs `forgetting` + mixed-effects 或回归线阴影。
- 附图：§4.3 分层子集示意（**标注 exploratory / 小 n**）。
- 附录：策略 Pareto + 配对检验表；SVCCA/CKA **协议伪代码**一页。

## 刻意不做（防范围膨胀）

- 不再新增冻结策略名称；不把 `spectrum_freeze` 写成主贡献。
- 不在未做敏感性前，把 SVCCA/CKA 写成「全体方法族失效」。

## 仅诊断、不训练

```powershell
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" smoke\run_compat_scan.py --config smoke\configs\real_smoke_qwen15b_8pairs_quick.yaml --output-dir outputs\real_smoke_qwen15b_8pairs_quick\compat_scan
```

---

## AutoDL / 云服务器（7B 跑数）最低建议

- **GPU 显存**：优先 **≥24GB**（如 RTX 3090/4090、A10）；7B fp16 权重 + LoRA + 激活与缓存，16GB 易 OOM 或需大幅减 `max_seq_len` / `max_train_samples`。  
- **磁盘**：系统盘外建议 **≥50GB** 空闲（模型快照 + `datasets` 缓存 + 多 seed 输出）；可将 `HF_HOME`/`HF_DATASETS_CACHE` 指到数据盘。  
- **软件**：与本地对齐即可——**Python 3.10+**、`pip install -r requirements.txt`、**CUDA 与 torch 版本匹配**（AutoDL 镜像通常已配好）。  
- **同步代码**：`git clone https://github.com/keyiadiannao/Lora-forgot.git`，在服务器改 yaml 里 `model.local_path` 或先 `huggingface-cli download` 再填路径；**勿把密码或 token 写进仓库**。  
- **安全**：不要在聊天/ issue 里发 **SSH 密码**；用 **SSH 公钥登录**、仓库用 **HTTPS+PAT** 或 **SSH deploy key**。
