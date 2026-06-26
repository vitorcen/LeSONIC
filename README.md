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

## 🥋🏃💃 演示 · flow3 LAFAN 技能 / Demo — flow3 LAFAN skills

从 LAFAN clip **按秒手挑**干净动作窗口 → 过**物理可行性筛查**（关掉严格安全包络，只看机器人**到底有没有真摔**：10 个窗口里严格闸门只过 2 个，但 6 个从头到尾没摔）→ 录 token 训进**同一个 VLA** → 拼成 flow3 循环：
**防守-蹬腿 → 转圈跑 → 搏击-连踢 → 快跑-倒跑 → 太空漫步 → 慢跑-倒跑**。
_Hand-picked LAFAN windows pass a **physical-validity screen** (“did it actually fall?”, not “did a strict safety threshold trip?” → 6/10 never fall vs 2/10 under strict), recorded into the **same** VLA, looped as flow3 (fight / run / dance)._

https://github.com/user-attachments/assets/add401c4-4628-4713-b82b-ab8b9b664a50

> 原理（token=「什么」/ WBC=「怎么」、**为何仍需参考 `.pkl`**、数据集 6 步管线、**两道筛查闸门**）：[`doc/sonic_vla_principles.html`](doc/sonic_vla_principles.html)。
> 跑法 `bash scripts/gear_sonic_flow3.sh`（🔁 离线回放，**保证流畅**）/ `bash scripts/gear_sonic_live_demo.sh @flow3`（🛰️ live 实时推理）。逐窗口筛选见 `LAFAN.ipynb`。

## 🧱 路线 B · MaskBeT — 从零小模型当 token 生产者 / Route B, from-scratch token producer

同一条 SONIC 链路，把 token 生产者从 GR00T（2B VLM）换成 **[MaskBeT](https://github.com/vitorcen/MaskBeT)**
—— 一个**无 backbone、从零 25M** 的 masked transformer（MaskGIT 并行解码 40×78 FSQ token 网格），
作为 `LeSONIC/MaskBeT` submodule。目的：不靠 VLM 验证「差距来自数据还是骨干」。
_Swap the token producer for a backbone-free, from-scratch 25M masked transformer — same SONIC WBC wire._

**flow3 8 窗口开环 MSE-64（同记忆口径）**：

| 生产者 / producer | expected | argmax | 备注 |
|---|---|---|---|
| GR00T N1.7（2B VLM + 动作预训练） | — | **0.0026** | 天花板 |
| **MaskBeT 25M（从零）** | **0.0090** | 0.0193 | expected 反超 CE best |
| StarVLA CE v1（4B 冻结 VLM） | 0.0125 | 0.0174 | 前 best |
| per-window-mean 模板 | — | 0.0367 | 下限 |

> **结论**：25M 从零 ≈/优于 4B 冻结 VLM（argmax 打平、expected 反超）→ **瓶颈是数据 + obs，不是骨干**。
> 闭环目检（RELAX 下 flow3 六段循环）幅度比 CE 足（iterative argmax 保 token std ≈ GT）。
> 都是记忆口径（无 held-out），held-out 泛化是下一步。设计/数据/尺寸/外审：[`MaskBeT/doc/maskbet_design.html`](MaskBeT/doc/maskbet_design.html)。
> 跑法 `bash scripts/maskbet_sonic_live_demo.sh @flow3`（server :5557，Isaac 侧零改动；见 `LAFAN.ipynb`）。

## 🤗 发布物 / Published

| | |
|---|---|
| 🌟 模型 V2 / Model V2 | [`wsagi/GR00T-N1.7-G1-SONIC-BonesSeed-V2`](https://huggingface.co/wsagi/GR00T-N1.7-G1-SONIC-BonesSeed-V2) — **冷启动修复版**（物理增广 + onset 加权 + stand，kick/walk/jump 可从冷站立自启，零 bootstrap）+ 2 段闭环 demo |
| 🌟 数据集 V2 / Dataset V2 | [`wsagi/SONIC-VLA-BonesSeed-V2`](https://huggingface.co/datasets/wsagi/SONIC-VLA-BonesSeed-V2) — LeRobot v2.1，54 ep / 12 630 帧（7 动作 + **物理 WBC-rollout 过渡增广** + stand） |
| 模型 V1 / Model V1 | [`wsagi/GR00T-N1.7-G1-SONIC-BonesSeed`](https://huggingface.co/wsagi/GR00T-N1.7-G1-SONIC-BonesSeed) — baseline checkpoint-8000（一次性动作需 bootstrap）+ 7 闭环 demo |
| 数据集 V1 / Dataset V1 | [`wsagi/SONIC-VLA-BonesSeed`](https://huggingface.co/datasets/wsagi/SONIC-VLA-BonesSeed) — LeRobot v2.1，7 动作各 1 ep（3815 帧） |
| 血缘 / Lineage | `bones-studio/seed → nvidia/GEAR-SONIC → wsagi/SONIC-VLA-BonesSeed{,-V2} → wsagi/GR00T-N1.7-G1-SONIC-BonesSeed{,-V2}` |

> ▶️ **直接跑(没自己训练)**：`BonesSeed.ipynb` 顶部 ⬇️ 下载 cell 会拉 V2，1.1/1.2/1.2a 自动解析「自训 → 否则 V2」。
> ⚠️ **状态 = derisk / proof-of-concept**：记忆 8 prompt，**泛化未测**；闭环 demo 在放宽终止（`RELAX=1`）下录制；
> V2 冷启动 LAUNCH 仍有随机性（每动作 ~50–100%，非确定 3/3）。完整批判 + roadmap 见 [`doc/sonic_vla_critique_roadmap.html`](doc/sonic_vla_critique_roadmap.html)。

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
│   ├─ serve_maskbet_sonic.py       # 路线 B：MaskBeT token server（同 ZMQ wire，:5557）
│   ├─ maskbet_sonic_live_demo.sh   # 路线 B：一键 live demo（MaskBeT → SONIC WBC）
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
│   ├─ sonic_vla_principles.html         # 🆕 原理 + 数据集 6 步 + 两道闸门 + 为何仍需 pkl
│   ├─ sonic_dance_motion_source.html    # 动作源三路对比
│   ├─ sonic_vla_closeloop_validation.html# 闭环验证（无记忆策略本质 + bootstrap）
│   └─ sonic_vla_critique_roadmap.html    # 🔬 三模型联合评审 + P0–P4 roadmap
├─ SONIC.ipynb                     # 全流程一键 notebook（置根；③区WBC原生 / ④区自训驱动）
├─ MaskBeT/                        # 路线 B submodule (vitorcen/MaskBeT)——从零 25M masked transformer
├─ dependencies/                   # 自包含 submodule（clone 用 --recursive）
│   ├─ Isaac-GR00T/               # 嵌套 submodule (NVIDIA/Isaac-GR00T, N1.7)——LeSONIC 自有副本，与父 PickOrange 互不干扰
│   ├─ GR00T-WholeBodyControl/    # 嵌套 submodule (NVlabs)，GEAR-SONIC WBC（SONIC 专属）
│   └─ starVLA/                   # 嵌套 submodule (vitorcen/StarVLA fork, 分支 main)——A/B+P0 引擎；patch 已打进 fork
├─ datasets/  (gitignored runtime) # sonic_vla_{raw,lerobot,pred_8k_final}
└─ outputs/   (gitignored runtime) # gr00t_sonic_8k（ckpt）
```

> **自包含**：LeSONIC 自带 `dependencies/{Isaac-GR00T,GR00T-WholeBodyControl,starVLA}` 三个嵌套 submodule —— **与父 `isaaclab-experience` 完全独立**（各自 checkout + `.venv`，分权好维护）。starVLA 指向 [`vitorcen/StarVLA`](https://github.com/vitorcen/StarVLA) fork 的 `starVLA_dev` 分支，本地改动（含 SONIC 专属 `QwenPI_CE` head + proprio-history）已作为 commit 打进 fork，`--recursive` clone 即得，**无需 apply patch**；后续维护准则见 fork 内 `CLAUDE.md`。脚本经 `$REPO_ROOT`（=LeSONIC 根，`scripts/..`）自解析,所有依赖/数据/产物都在 LeSONIC 内。
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
