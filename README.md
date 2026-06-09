# LeSONIC — GR00T × GEAR-SONIC 全身动作 VLA（架构侧路线）

Prompt 驱动 Unitree G1 全身动作的 **架构侧（architecture-side）** 路线：一个 **GR00T N1.7** VLA 只输出
[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 全身控制器（WBC）的 **64 维 FSQ motion token**，
SONIC WBC 当现成「平衡底座」把 token 解码成 29-DoF 关节动作。GR00T 只学 `prompt → token`，平衡/recovery 由预训练 WBC 兜底。

> `prompt + ego-cam + proprio → GR00T N1.7 →(ZMQ)→ 64-d token → SONIC.decode(50Hz) → 29-DoF G1`
> _One prompt-conditioned model, skills live in the **token space** — not behavior-cloned per joint._

## 🎬 演示 · 动作串联 / Demo — action sequencing

一句话**按时序切换 prompt**，G1 在**一个运行会话**里连贯做 蹲 → 走 → 跳舞 → … 并循环，**不重启 GUI/server**，
**走到新位置就在那里作下一个动作**。这是对记忆动作的 **prompt 编排**，非复合指令理解。
_Prompt switched over time → the G1 chains motions and loops in one live session; it walks to a new spot and performs the next action there._

https://github.com/user-attachments/assets/8f212cba-f8f0-46bc-8483-cf5d704e2bce

> 原理 + 三时钟调试（`_resample_command` 在 clip 边界硬传送机器人绕过 `dones`）+ `SONIC_NO_REF_RESAMPLE` freeze 修复：
> [`doc/sonic_action_sequencing.html`](doc/sonic_action_sequencing.html)。跑法 `bash scripts/gear_sonic_live_demo.sh @flow2`（`list`/`@flow1`/即兴序列）。

## 🤗 发布物 / Published

| | |
|---|---|
| 模型 / Model | [`wsagi/GR00T-N1.7-G1-SONIC`](https://huggingface.co/wsagi/GR00T-N1.7-G1-SONIC) — checkpoint-8000 推理权重 + 7 闭环 demo |
| 数据集 / Dataset | [`wsagi/SONIC-VLA-LeRobot`](https://huggingface.co/datasets/wsagi/SONIC-VLA-LeRobot) — LeRobot v2.1，7 动作各 1 ep（3815 帧） |
| 血缘 / Lineage | `bones-studio/seed → nvidia/GEAR-SONIC → wsagi/SONIC-VLA-LeRobot → wsagi/GR00T-N1.7-G1-SONIC` |

> ⚠️ **状态 = derisk / proof-of-concept**：记忆 7 动作（各 1 ep），**泛化未测**；闭环 demo 在放宽终止（`RELAX=1`）下录制。
> 完整批判 + P0–P4 roadmap 见 [`doc/sonic_vla_critique_roadmap.html`](doc/sonic_vla_critique_roadmap.html)（三模型联合评审）。

## 目录 / Layout

```
LeSONIC/
├─ scripts/                         # 全流程脚手架（路径自解析到仓库根）
│   ├─ gear_sonic_setup.sh          # 一次性装 GEAR-SONIC WBC（submodule + apply patches）
│   ├─ gear_sonic_preview.sh        # 单进程 Isaac-eval 预览 WBC 原生动作
│   ├─ gear_sonic_preview_sim2sim.sh# 双进程 MuJoCo sim2sim（部署忠实，重前置）
│   ├─ gear_sonic_demo.sh           # WBC 原生动作一键 demo
│   ├─ gear_sonic_record.sh         # ① 录 (token, ego_rgb, proprio) → datasets/sonic_vla_raw
│   ├─ sonic_vla_npz_to_lerobot.py  # ② npz → LeRobot v2.1 数据集
│   ├─ gr00t_sonic_finetune.sh      # ③ finetune GR00T-N1.7（冻 VLM 训 DiT head；SMOKE=1 验证）
│   ├─ gr00t_dump_pred_tokens.py    # ④ open-loop dump 预测 token
│   ├─ gear_sonic_inject.sh         # 离线 token 回放注入（RELAX=1 默认）
│   ├─ gear_sonic_live.sh           # ✅ 闭环：live GR00T 在环驱动 WBC（BOOTSTRAP/RELAX 可调）
│   ├─ gr00t_resume_heal.sh         # 自愈 resume（删损坏 ckpt + 续训，扛 mid-save 崩）
│   ├─ gr00t_keep_stage_ckpts.sh    # 阶段 ckpt hardlink 进 keep/（防 HF 滚删，零额外磁盘）
│   ├─ gr00t_ckpt_intact.py         # 校验 safetensors 完整（resume 用）
│   ├─ gr00t_resave_bf16.py         # P0#3 fp32→bf16 重存（纯 CPU，12.6→6.3G）
│   ├─ make_holdout_split.py        # P0#2 leave-one-motion-out held-out split（go/no-go 闸门）
│   ├─ apply_gear_sonic_patches.sh  # 幂等 apply WBC 补丁
│   └─ apply_gr00t_n17_patches.sh   # 幂等 apply Isaac-GR00T 补丁
├─ patches/
│   ├─ gear-sonic/                  # 0001 download-symlink / 0002 textured-usd / 0003 recorder+injector / 0004 live-injector
│   └─ gr00t-n17/                   # 0001 finetune OOM fix（adafactor + grad-ckpt）
├─ doc/                            # 单文件 HTML（内嵌 SVG，中英对照）
│   ├─ groot_sonic_wbc_route.html        # 架构总览 Stage A/B/C + derisk 复盘
│   ├─ sonic_dance_motion_source.html    # 动作源三路对比
│   ├─ sonic_vla_closeloop_validation.html# 闭环验证（无记忆策略本质 + bootstrap）
│   └─ sonic_vla_critique_roadmap.html    # 🔬 三模型联合评审 + P0–P4 roadmap
├─ SONIC.ipynb                     # 全流程一键 notebook（置根；③区WBC原生 / ④区自训驱动）
├─ dependencies/                   # 自包含 submodule（clone 用 --recursive）
│   ├─ Isaac-GR00T/               # 嵌套 submodule (NVIDIA/Isaac-GR00T, N1.7)——LeSONIC 自有副本，与父 PickOrange 互不干扰
│   └─ GR00T-WholeBodyControl/    # 嵌套 submodule (NVlabs)，GEAR-SONIC WBC（SONIC 专属）
├─ datasets/  (gitignored runtime) # sonic_vla_{raw,lerobot,pred_8k_final}
└─ outputs/   (gitignored runtime) # gr00t_sonic_8k（ckpt）
```

> **自包含**：LeSONIC 自带 `dependencies/{Isaac-GR00T,GR00T-WholeBodyControl}` 两个嵌套 submodule —— **与父 `isaaclab-experience` 的 Isaac-GR00T 完全独立**（各自 checkout + `.venv`，分权好维护）。脚本经 `$REPO_ROOT`（=LeSONIC 根，`scripts/..`）自解析,所有依赖/数据/产物都在 LeSONIC 内。
> clone：`git clone --recursive git@github.com:vitorcen/LeSONIC.git`（或 `git submodule update --init --recursive`）。`.venv` 不入 git，每个 submodule 各建（GR00T 训练栈 torch2.7.1+cu128；WBC 走 isaaclab + `.venv_data_collection`）。

## 快速上手 / Quickstart

```bash
# 0. 装 WBC（submodule + patches）
bash LeSONIC/scripts/gear_sonic_setup.sh

# 1. 闭环跑（先起 GR00T ZMQ server，再驱动 WBC）—— 见 model card「How to run」
python -m gr00t.eval.run_gr00t_server --model_path <ckpt> --embodiment_tag unitree_g1_sonic --port 5555
bash LeSONIC/scripts/gear_sonic_live.sh macarena            # 自持动作：纯 live
BOOTSTRAP=80 bash LeSONIC/scripts/gear_sonic_live.sh kick   # 一次性动作：bootstrap 触发后交 live
RELAX=0 bash LeSONIC/scripts/gear_sonic_live.sh macarena    # 严格终止（真实稳定性测试）

# 自训全流程：record → convert → finetune → dump → inject（见 SONIC.ipynb ④区）
```

## 已知边界 / Known limits（详见评审 doc）

- **无记忆单帧策略**：单帧 obs→40-step token，无 history/phase → 一次性动作（kick/walk/jump）从静止启不动，需 bootstrap。**架构性，加数据治不了**。
- **无 held-out**：MSE 0.0011 是训练集记忆误差，非泛化。
- **「不摔」≠ VLA 功劳**：是 SONIC WBC 托底；demo 在 RELAX=1（终止关闭）下录。
- **FSQ 离散码当连续回归** / **ego 相机看地面视觉弱** / **依赖外部 WBC + ZMQ 跨进程**。

### P0 进度（2026-06-08）

| # | 修复 | 状态 |
|---|---|---|
| #3 | F32→bf16 重存（12.6→6.3G） | ✅ 完成 `scripts/gr00t_resave_bf16.py` → `outputs/gr00t_sonic_8k_bf16/`（待上传） |
| #1 | FSQ snap-to-grid（推理侧，免重训） | ✅ 完成 两 injector + patch，env `SONIC_SNAP_GRID`（默认开）；实测 GR00T 预测 **60% 维度 off-grid、值域出 [-0.5,0.5] 界**，snap 把它拉回 decoder 唯一见过的网格。**闭环增益待 GPU A/B** |
| #2 | held-out split（go/no-go 闸门） | ✅ 切分工具完成 `scripts/make_holdout_split.py`（LOMO，已验证）；**重训+held-out eval 待 GPU** |
| #1' | loss 改 per-dim CE（训练侧） | ⏳ 需重训（GPU）；snap 是其推理侧近似 |
| #4 | 页面诚实化 | 📝 草稿就绪 `/tmp/sonic_card_revised.md`，**待确认上传 HF** |

> snap-to-grid 用法：`SONIC_SNAP_GRID=0 bash scripts/gear_sonic_live.sh macarena` 关闭以做 A/B 对照。
完整 roadmap 见 `doc/sonic_vla_critique_roadmap.html`。
