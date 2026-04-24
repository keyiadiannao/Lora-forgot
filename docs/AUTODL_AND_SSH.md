# AutoDL 与 SSH 工作备忘

本文记录本机到 AutoDL 实例的常用连接方式、目录约定与实验命令，避免换机或隔段时间后遗忘。

---

## 1. 本机（Windows）SSH 配置

### 1.1 配置文件位置

- OpenSSH 客户端配置：`%USERPROFILE%\.ssh\config`
- 本仓库关联实例使用的私钥（示例名）：`%USERPROFILE%\.ssh\autodl_ed25519`  
  - **私钥不要提交到 Git**，不要贴到聊天或截图里。

### 1.2 推荐：`Host` 别名（已在本机配置时使用）

在 `config` 中为该实例增加一段（端口、主机名以 AutoDL 控制台「SSH 登录」为准，实例重建后会变）：

```sshconfig
Host autodl-c48
  HostName connect.westd.seetacloud.com
  Port 19697
  User root
  IdentityFile C:\Users\<你的用户名>\.ssh\autodl_ed25519
  IdentitiesOnly yes
```

日常使用：

```powershell
ssh autodl-c48
```

### 1.3 备用：命令行显式指定（无 config 或换机时）

```powershell
ssh -p 19697 -i $env:USERPROFILE\.ssh\autodl_ed25519 -o IdentitiesOnly=yes root@connect.westd.seetacloud.com
```

若出现 `Permission denied (publickey)`：检查控制台端口/域名是否已更新、私钥路径是否正确、`IdentitiesOnly yes` 是否避免误用其它密钥。

### 1.4 同步代码到服务器

在仓库根目录（示例）：

```powershell
scp smoke\run_smoke.py smoke\run_multiseed.py autodl-c48:/root/work/Lora-forgot/smoke/
```

多文件可一行写多个源路径；目标目录需已存在。

---

## 2. 服务器侧约定

| 项 | 路径或说明 |
|----|------------|
| 仓库根目录 | `/root/work/Lora-forgot` |
| Conda Python | `/root/miniconda3/bin/python` |
| 大缓存（HF / torch） | 常用前缀 `/root/autodl-tmp/hf/`、`/root/autodl-tmp/torch` |
| 后台实验日志（自定义） | `/root/autodl-tmp/logs/` |

运行前建议在 shell 里导出（与现有脚本一致即可）：

```bash
export HF_HOME=/root/autodl-tmp/hf/home
export HF_HUB_CACHE=/root/autodl-tmp/hf/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers
export TORCH_HOME=/root/autodl-tmp/torch
```

---

## 3. 多 seed 实验（含 k2 列重跑）

- 入口：`smoke/run_multiseed.py`（顺序跑各 seed 的 `run_smoke.py --mode real`，再聚合）。
- Linux 上 `--run-smoke-path` 请用正斜杠：`smoke/run_smoke.py`。

本地仓库中的流水线示例（上传后可在服务器上 `nohup bash ...` 执行）：

- `smoke/scripts/autodl_rerun_k2_pipeline.sh`

服务器上若已拷贝到日志目录，例如：

- 脚本：`/root/autodl-tmp/logs/autodl_rerun_k2_pipeline.sh`
- 聚合日志：`/root/autodl-tmp/logs/rerun_k2_pipeline.nohup.log`
- 父进程 PID 可手写保存到：`/root/autodl-tmp/logs/rerun_k2_pipeline.pid`

查看进度：

```bash
ssh autodl-c48 'tail -f /root/autodl-tmp/logs/rerun_k2_pipeline.nohup.log'
```

检查是否仍占用 GPU：

```bash
ssh autodl-c48 'nvidia-smi; pgrep -af run_smoke'
```

跑完后在对应 `outputs/.../multiseed_pair_metrics.csv` 中应出现列 `activation_principal_cos_k2`（需用含 k2 的 `run_smoke.py` 重跑各 seed 后才会写入）。

---

## 4. 实例重建后需要更新的项

AutoDL 关机/换卡后常见变化：

- SSH **端口**、有时 **HostName**（仍以控制台为准）。
- `~/.ssh/config` 里对应 `Host` 的 `Port` / `HostName`。
- 若数据盘路径变化，检查 yaml 里 `model.local_path`、`hf_datasets_cache` 等是否仍指向当前实例上的缓存目录。

---

## 5. 相关仓库文件

- 协议与指标说明：`docs/SCALING_PROTOCOL.md`
- 执行计划与数据路径索引：`NEXT_STEPS.md`
- AutoDL 专用 yaml：`smoke/configs/*_autodl_local.yaml`
