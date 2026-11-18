# ImageWAM Geometry：SIMPLE RGB-D 四组实验

本仓库基于 [ImageWAM](https://github.com/yuyangalin/ImageWAM)，提供适配 SIMPLE RGB-D 的训练代码，不是上游官方发布。当前默认分支使用 **action chunk 32、相对旋转监督、全局 XYZ 监督**，**不启用相机相对平移预测**，也没有局部几何 tokens。

这份 README 可以直接交给 Codex 按步骤执行。文中的命令在 **Linux 训练服务器、仓库根目录**运行；Windows 用户先 SSH 到训练服务器。正式训练必须使用已经分配且空闲的 GPU。

## 1. 四个实验分别是什么

四组都训练图像和动作，使用相同数据、初始化权重、chunk、学习率计划和全局 batch。

| 实验 | 启动名称 | 配置 | 旋转 tokens / 头 | 几何 tokens / Depth Pro 头 | 总损失 |
|---|---|---|---|---|---|
| baseline | `baseline` | `imagewam_baseline` | 无 | 无 | `0.5 L_image + L_action` |
| only pose | `only_pose` | `imagewam_only_pose` | 32 个，有 | 无 | `0.5 L_image + L_action + 5 L_rotation` |
| only depth | `only_depth` | `imagewam_only_depth` | 无 | 220 个，有 | `0.5 L_image + L_action + 0.5 L_XYZ` |
| both / 完整 geometry | `both` | `imagewam_both` | 32 个，有 | 220 个，有 | `0.5 L_image + L_action + 0.5 L_XYZ + 5 L_rotation` |

**这里的 pose 仅指相对旋转，不包含平移。** `geometry` / `task=imagewam_geometry` 是 both 的兼容入口。`only_pose` 和 `only_depth` 会真正移除对应 tokens、预测头和损失，不是仅把 loss 权重置零。

baseline 使用原 ImageWAM 主干，不创建辅助模块。其余三组在同一主干上开启对应分支；全部 25 层的条件注意力连接保留。四组共用尾部采样规则。为了统一数据管线，加载器仍可读取辅助 GT，但关闭的分支不会使用它计算预测或损失。

### 模型与监督约定

- 图像主干：FLUX.2 **klein-base-4B**；动作主干：ActionDiT，动作维度 36，机器人 state 维度 32。
- 四组实际预测长度均为 32。底层 `max_action_horizon: 64` 是可容纳的最大长度，不是本次采样窗口或动作 token 数量。
- 当前文本特征、当前 RGB 和当前机器人 state 是条件；相机位姿和深度 GT 只作为训练标签。
- RGB 输入 **176×320**；全局深度／XYZ 输出 **256×448**。原始数据为 360×640，加载器按输出尺寸缩放相机内参。
- 全局几何 tokens 为 11×20=220 个，取第 5、10、15、20、25 层状态，经多尺度适配层和 Depth Pro 解码器预测米制深度，再用内参反投影得到 XYZ。
- `L_XYZ` 是未来相机坐标系中的 XYZ Smooth L1，`beta=0.1`；GT 来自仿真米制深度。不是把世界坐标位置作为输入。
- 旋转头输出连续 6D 表示，再转换成旋转矩阵；`L_rotation = ||R_pred - R_gt||_F² / 8`。日志中的角度误差单位是度，和训练 loss 的数值不同。
- 图像、动作使用各自 flow-matching 目标的 MSE，并保留原 scheduler 权重；不是直接对 RGB 像素做 MSE。
- 原图文和动作 token 的时间调制沿用 baseline；辅助 tokens 使用零时刻调制。没有 geometry 额外的整层激活重计算，原有注意力和深度解码器的 checkpoint 策略保留。

### both 的实际注意力关系与损失

记 `C` 为当前文本、RGB 和 state 条件，`P` 为 FLUX.2 侧的 32 个旋转 tokens，`F` 为训练时加噪的未来图像 tokens，`G` 为 ActionDiT 侧的 220 个全局几何 tokens，`A` 为 32 个加噪动作 tokens。state 编码后拼在有效文本条件后，不作为一个独立的 ActionDiT 输入分支。

下表每行表示该组查询可以读取哪些组的 K/V；这是全部 25 层使用的关系，并非按表格顺序串行执行：

| 查询 tokens | 可读取的 K/V |
|---|---|
| C | C |
| P | C、P |
| F | C、P、F |
| G | C、P、G |
| A | C、P、G、A |

因此动作可读取几何，几何不能反读动作；P、G、A 都不能直接读取 F。全部层参与条件注意力，但深度头只取上述五层 G 的状态。P、G 是可学习寄存 tokens，不加扩散噪声；图像和动作分支分别学习 flow-matching 去噪。

Depth Pro 只复用预训练解码器和深度头，未使用其图像编码器。几何适配层把 G 转成多尺度特征；解码器可训练，学习率为主学习率的 0.1 倍。深度经过平滑正值转换，再按几何输出分辨率对应的内参反投影：`XYZ = Z × K⁻¹[u+0.5, v+0.5, 1]ᵀ`。这里是未来端点相机坐标系中的 XYZ，不需要预测旋转来变换到当前帧，也没有额外的 Z 损失。

`L_both = 0.5 L_image + L_action + 0.5 L_XYZ + 5 L_rotation`。XYZ 对三个坐标分量取 Smooth L1 平均，再对每个样本的有效像素平均；旋转监督 `R_tᵀ R_(t+i)`（i=1…32），对每个样本的有效未来时刻平均。两项再在有有效标签的样本间平均。补齐动作、无效深度和缺失未来旋转不参与相应损失；图像、动作仍保留原有 scheduler 的训练权重。日志中的 `loss_relative_pose` 在当前配置下等于旋转损失，不是额外加一项，也不包含平移。

### chunk 32 和尾部处理

窗口以当前帧 `t` 为条件，监督动作 `t ... t+31`，未来图像与 XYZ 对应 `t+32`；32 个旋转标签分别是 `t+1 ... t+32` 相对 `t` 的旋转。

- 不足 32 个动作的真实尾部仍保留；仅补齐位置被 `action_is_pad` 屏蔽。
- 旋转单独按 `geometry_rotation_valid` 监督现存的有效未来帧，按有效数量平均。
- 不存在真实 `t+32` 时，关闭这个样本的未来图像和 XYZ loss，不把当前帧占位图当成真实未来。
- 最后一帧若有有效动作，可以监督该动作；没有后续观测则不监督旋转。无效观测、内部断裂和人工尾帧不会被当作真实未来。

## 2. 当前数据版本，以及从旧版迁移

数据地址：[ivanhe666/simple_rgbd](https://modelscope.cn/datasets/ivanhe666/simple_rgbd/tree/master/SIMPLE-RGBD-Teleop)。为避免 `master` 后续变化影响复现，本代码固定下载：

| 项目 | 当前固定值 |
|---|---|
| ModelScope revision | `1aa446155d6597e91169872a965557640fb31871` |
| 轨迹数 | **789** |
| 总帧数 | **312,139** |
| 任务数 | **8** |
| chunk 32 有效动作采样窗口 | **311,072** |
| `meta/episodes.jsonl` SHA256 | `101608685679811fc4c97fa05b14a34578fd886887e4773dea6b3a13c8d2f9a1` |

相比旧的 792 条 / 313,553 帧版本，这次删除了**旧编号** 37、398、425 三条 OpenTrashCan 轨迹，并重新连续编号；现在这些编号可能对应其他轨迹，**不要再次按编号删除**。数据主结构和字段不变，视频仍是 `.mkv`。对应缓存、索引和归一化统计也已更新，必须与数据一起使用。

| 任务 | 条数 |
|---|---:|
| Handover | 75 |
| LocomotionPickBetweenTables | 85 |
| OpenFaucet | 95 |
| OpenOven | 97 |
| OpenTrashCan | 49 |
| PickAndPlaceAndHugContainer | 159 |
| PushOfficeChair | 85 |
| XMoveBendPick | 144 |

版本信息集中在 `configs/dataset_release.json`。检查程序同时验证数据清单、info、训练清单和归一化统计的哈希，不能只改轨迹数量来绕过版本检查。

**已有旧数据时下载到新目录。** 默认新目录是 `data/simple_rgbd_1aa44615/`，不会主动覆盖原来的 `data/SIMPLE-RGBD-Teleop/`。下载脚本发现目标目录中的关键文件属于其他版本会停止。训练期间不要原地替换数据，不要混用旧缓存，也不要将旧数据的完整训练 state 直接恢复到重新编号后的数据上。

## 3. 第一步：克隆代码并检查服务器

仓库若为私有，先接受协作者邀请，使用自己的 GitHub 登录或 SSH 密钥。

```bash
git clone https://github.com/Ivan-hexy/imagewam_geometry.git
cd imagewam_geometry
git switch main
git rev-parse HEAD
nvidia-smi
df -h .
gcc --version
```

已有仓库先检查本地改动：

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only
```

如果有未提交改动或无法 fast-forward，先确认这些改动的用途，不要用 `reset --hard` 覆盖。当前流程使用 `main`，不要切换到包含相对平移试验的 `pose-metric-pilot`。

准备 Linux、NVIDIA 驱动、Git、Python 3.11 和 uv。PyTorch wheel 使用 CUDA 11.8；DeepSpeed 扩展可能还需要服务器 CUDA toolkit、C++ 编译器和 Ninja。数据约需 200 GB，另外为权重、环境及四组检查点预留空间；四组完整训练 state 和多个权重文件会再占用数百 GB，先检查磁盘或账户配额。

## 4. 第二步：创建训练环境

```bash
# 如果服务器尚无 uv，先安装；已有则跳过这一行。
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[shared]'
cp -n .env.example .env.local
source scripts/env.sh
python -c 'import torch, av, h5py, deepspeed; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

已使用的环境组合包括 Python 3.11、PyTorch 2.7.1/CUDA 11.8、Transformers 4.56.1、Accelerate 1.12.0、DeepSpeed 0.18.5。依赖版本见 `pyproject.toml`。`third_party/` 已带有必要的 FLUX.2 和 Depth Pro 源码；不要额外安装它们整套项目依赖覆盖当前 PyTorch。

每次新开终端后，先进入仓库，再执行：

```bash
source .venv/bin/activate
source scripts/env.sh
```

## 5. 第三步：下载数据与配套缓存

ModelScope 下载工具使用独立环境，避免改变训练依赖：

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install modelscope-hub==0.7.1
# 若仓库访问要求登录，使用自己的魔搭账号完成此步骤：
.download-venv/bin/ms-hub login
bash scripts/download_data.sh
```

GitHub、魔搭、Hugging Face 的权限和 Token 是三套独立凭据。不要将凭据写入代码、README 或 Git 远程 URL；下载报 401/403 时检查对应平台账号权限。

下载脚本等价于以下命令，两种方式选一种：

```bash
.download-venv/bin/ms-hub download ivanhe666/simple_rgbd \
  --repo-type dataset \
  --revision 1aa446155d6597e91169872a965557640fb31871 \
  --local-dir ./data/simple_rgbd_1aa44615 \
  --max-workers 4
```

预期目录：

```text
imagewam_geometry/
  data/simple_rgbd_1aa44615/
    SIMPLE-RGBD-Teleop/
      meta/episodes.jsonl
      meta/info.json
      data/...                    # 动作、state 等 parquet
      videos/...                  # RGB MKV
      depth/...                   # 米制深度、内外参、动作与有效掩码 HDF5
      trajectories/...
    SIMPLE-RGBD-Teleop-cache/
      all_verified_stats.json
      all_verified_training_manifest.json
      qwen3/                      # 预计算文本特征
```

不要只下载 RGB，不需要重新渲染、转换视频或重算这套已匹配的缓存。

### 数据放到其他磁盘／复用已有新版数据

```bash
DATA_DOWNLOAD_DIR=/mnt/datasets/simple_rgbd_1aa44615 bash scripts/download_data.sh
```

然后编辑 `.env.local`，同时设置两个目录，保留文件中已有的其他配置：

```bash
export SIMPLE_RGBD_ROOT=/mnt/datasets/simple_rgbd_1aa44615/SIMPLE-RGBD-Teleop
export SIMPLE_RGBD_CACHE=/mnt/datasets/simple_rgbd_1aa44615/SIMPLE-RGBD-Teleop-cache
```

```bash
source scripts/env.sh
```

`DATA_DOWNLOAD_DIR` 只改变下载目的地，不会自动修改训练路径。已有同版本完整数据可跳过下载，直接配置上述变量并执行下一节的检查。

## 6. 第四步：下载模型权重并生成 ActionDiT 初始化

### 推荐：直接下载已准备好的魔搭初始化包

公开模型仓库：[ivanhe666/imagewam_pretrained](https://modelscope.cn/models/ivanhe666/imagewam_pretrained)。四份权重合计约 **12.38 GB**，包含已经生成的 `action_dit_init.pt`，可供四组实验共用；这不是训练后的机器人策略检查点。

在本代码仓库根目录执行，沿用第三步的 `.download-venv`：

```bash
.download-venv/bin/ms-hub download ivanhe666/imagewam_pretrained \
  --repo-type model \
  --revision f4f1ff471c36b13055c537d74a7a71e741dfc551 \
  --local-dir ./checkpoints --max-workers 4
(cd checkpoints && sha256sum -c SHA256SUMS)
source scripts/env.sh
```

四项校验都应显示 `OK`。下载后的四个文件路径与下表“默认路径”完全对应，**随后直接进入第五步检查，无需再运行本节后面的官方权重下载或初始化生成命令**。如果 `.env.local` 中有旧的自定义权重路径，先将其更新为本次文件路径。

这套包复用已通过四组冒烟测试的权重。包内 AE 实际来自 **klein-base-4B 官方 Diffusers VAE**，已转成 BFL 键名格式和 BF16；`FLUX.2-dev` 是兼容代码的保存目录名，不能据此判断来源。ActionDiT 文件仅清理了来源元数据中的服务器绝对路径，全部参数 tensor 保持一致。文件大小、哈希和转换信息见 [configs/pretrained_release.json](configs/pretrained_release.json)；各组件的原始许可和研究用途限制保留在魔搭仓库内。

### 备选：从官方来源下载并生成初始化

以下方式适合自行准备原始权重，与上面的魔搭下载方式选择一种。官方 dev AE 与包内已转换的 BF16 AE 不保证文件字节相同；四组公平比较应始终使用同一套初始化文件。

| 权重 | 官方来源与文件 | 默认路径 | 需要的实验 |
|---|---|---|---|
| FLUX.2 主干 | [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/main)，`flux-2-klein-base-4b.safetensors` | `checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors` | 四组 |
| 图像 AE | [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main)，仅 `ae.safetensors` | `checkpoints/flux2/FLUX.2-dev/ae.safetensors` | 四组 |
| Depth Pro 解码器初始化 | [Apple 官方文件](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt) | `checkpoints/depth_pro/depth_pro.pt` | only depth、both |
| ActionDiT 初始化 | 本仓库从 FLUX.2 权重生成 | `checkpoints/action_dit_init.pt` | 四组 |

使用 **klein-base-4B**，不要换成蒸馏版 klein-4B、9B 或完整 dev 主干。ActionDiT 的统一默认文件名为 `action_dit_init.pt`，环境脚本、配置和初始化命令使用同一个路径；文件名本身不决定模型结构。

如 Hugging Face 页面要求接受条款或申请权限，先在对应模型页完成，再使用自己的账号登录：

```bash
source .venv/bin/activate
source scripts/env.sh
hf auth login
hf auth whoami
python scripts/download_weights.py --mode both
bash scripts/prepare_action_init.sh
```

跑四组只需下载一次。若仅准备 baseline 或 only pose，可把下载参数改成 `--mode baseline` 或 `--mode only_pose`，这样不下载 Depth Pro。ActionDiT 初始化脚本默认在 CPU 上转换一次；文件已经存在会跳过，转换时需要足够主机内存。

需要手动下载时，使用下面的命令替代下载脚本：

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B \
  flux-2-klein-base-4b.safetensors \
  --local-dir ./checkpoints/flux2/FLUX.2-klein-base-4B
hf download black-forest-labs/FLUX.2-dev ae.safetensors \
  --local-dir ./checkpoints/flux2/FLUX.2-dev
mkdir -p ./checkpoints/depth_pro
curl --fail --location --retry 3 \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  --output ./checkpoints/depth_pro/depth_pro.pt.download && \
mv ./checkpoints/depth_pro/depth_pro.pt.download ./checkpoints/depth_pro/depth_pro.pt
bash scripts/prepare_action_init.sh
```

如果复用其他位置的权重，编辑 `.env.local` 后重新 `source scripts/env.sh`：

```bash
export FLUX2_MODEL_PATH=/mnt/models/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/mnt/models/ae.safetensors
export SIMPLE_ACTION_INIT=/mnt/models/action_dit_init.pt
export DEPTH_PRO_CHECKPOINT=/mnt/models/depth_pro.pt
```

FLUX.2 下载脚本仍写入默认 `checkpoints/` 目录；上述变量控制加载位置。Depth Pro 下载脚本会使用 `DEPTH_PRO_CHECKPOINT` 指定的目的文件。

训练复用 `qwen3/` 中的文本缓存，无需另下载 Qwen3。在线处理新文本需要单独准备文本编码器。GitHub 不包含训练到 20k/30k/40k 的策略检查点；本节是从官方预训练模型开始训练所需的权重。

## 7. 第五步：正式启动前的检查

先检查四组配置及必要权重文件：

```bash
for mode in baseline only_pose only_depth both; do
  python scripts/check_setup.py --mode "$mode" --config-only --require-weights || exit 1
done
```

输出应全部为 `config_passed: true`、`translation_enabled: false`，tokens 数量与第一节表格一致。检查输出的 `paths` 和 `training`，确认指向本次数据、缓存和 `action_dit_init.pt`，而不是 shell 或旧 `.env.local` 留下的路径。`--config-only` 不读取数据；`--require-weights` 检查权重文件存在且非空，不代表完整加载模型已通过。

`train.sh` 启动前会使用同一组 Hydra 覆盖参数执行检查；手动核对覆盖参数时使用 `python scripts/check_setup.py --mode both --config-only --require-weights -- batch_size=8 gradient_accumulation_steps=2`。更改过默认路径的旧终端应退出重开，并更新 `.env.local` 后再加载环境。

再检查真实数据、缓存和全部采样窗口：

```bash
python scripts/check_setup.py --mode both --all-windows --require-weights
```

预期关键字段：

```json
{
  "episodes": 789,
  "frames": 312139,
  "actual_windows": 311072,
  "sampled_first_and_last_windows_passed": true,
  "translation_enabled": false
}
```

这一步核对四个关键元数据／缓存文件的哈希、全部轨迹引用文件存在性、全部窗口数量，并解码每个任务一条轨迹的首尾窗口。它不是对约 200 GB 数据逐文件重算 SHA256，也不是逐帧视觉质量审核。若不用 `--all-windows`，仅枚举抽样轨迹，`actual_windows` 为 null。

结构、梯度和完整训练冒烟已由代码提供方完成，受托训练人员无需重复这些测试；完成本机环境、权重和数据检查后，直接启动下面的四组正式实验。

## 8. 已完成的冒烟与测速结果（供受托人参考）

2026-11-18 已在同节点 **4 张 NVIDIA L40、每卡 bs1、梯度累计 1** 下执行：

| 实验 | 完整优化器更新 | 结果 |
|---|---:|---|
| baseline | 3 | 通过，loss 有限，无 OOM |
| only pose（仅旋转） | 3 | 通过，loss 有限，无 OOM |
| only depth | 3 | 通过，loss 有限，无 OOM |
| both | 3 | 通过，loss 有限，无 OOM |

记录见 [VALIDATION.json](VALIDATION.json)。测试没有 CPU 优化器卸载，没有相对平移监督。**无需受托人再跑 bs1 冒烟、200 步短测或结构测试。** 这些结果验证训练流程可运行，不代表 40k 收敛或机器人成功率。

### 四卡 bs1 更新速度

另做四组各 30 次更新，剔除前 5 次，统计第 6–30 次。统一同节点 4×L40、每卡 bs1、累计 1、每卡 8 workers；W&B 关闭，无 CPU 卸载。

| 实验 | 平均秒/更新 | 中位数秒/更新 | 相对 baseline |
|---|---:|---:|---:|
| baseline | 1.463 | 1.461 | 1.00× |
| only_pose | 1.866 | 1.866 | 1.28× |
| only_depth | 1.633 | 1.635 | 1.12× |
| both | 2.028 | 2.028 | 1.39× |

计时包含数据读取、前后向、多卡通信、优化器和日志，排除模型加载、前 5 次预热、评测和保存检查点。原始统计见 [SPEED_REPORT.json](SPEED_REPORT.json)。这是 bs1 的短时速度参考，不能直接推算正式全局 batch 64 的耗时；节点的 6、7 卡有其他用户任务。

W&B 已完成四组 SDK 离线记录检查，以及 both 在四卡 bs1 下的 3 次完整更新集成检查，正常写入指标并结束 run。网页项目已确认，但 L40 API 上传返回 403，云端曲线未验证成功，详见第 9 节；离线记录通过不等于在线上传成功。

## 9. 第七步：四组正式训练怎么启动

共同配置：**4 卡 × 每卡 batch 8 × 梯度累计 2 = 全局 batch 64**；每卡 8 workers；学习率计划和最大更新数均按 40,000 设置；峰值学习率 2.5e-5；warmup 1000 次后 cosine 衰减；每 10,000 次保存一次检查点。前 1000 次学习率上升是预热行为。

**本次实际执行采用配对提前停止：四组保留 `max_steps=40000`，每一对的两个实验都达到至少 30,000 次更新，且 30k 检查点保存完成后，停止这一对。** 不把 `max_steps` 改成 30,000，否则 cosine 衰减计划会改变。

| 执行批次 | 并行实验（各自同节点四卡） | 停止条件 | 统一评测检查点 |
|---|---|---|---|
| 第一对 | baseline ＋ only depth | 两者均达到至少 30k，且两份 30k 权重已完整保存 | 两者的 `step_030000.pt` |
| 第二对 | only pose ＋ both | 两者均达到至少 30k，且两份 30k 权重已完整保存 | 两者的 `step_030000.pt` |

同一对从各自的第 0 步同时启动；第一对结束并释放资源后，再启动第二对。较快实验可超过 30k；若先到 40k 则正常结束，继续等待较慢实验到 30k。最终四组统一用 **30k 权重**比较效果，不以不同步数的最后权重作公平对比。

当前训练脚本**没有配对自动停止控制器**，W&B 只监控，不负责终止作业。受托人负责监控两个 run 的 `train/optimizer_step`，并在服务器确认 `weights/step_030000.pt` 已完成写入、日志报告保存成功之后再停止相关训练进程／作业；仅看到曲线到 30k 或文件刚出现不够。避免在保存过程中强制杀进程。停止前记录两组实际步数，交付两份 30k 权重、配置、日志和 W&B 链接。超过 30k 的较快实验尚未保存的中间更新无需作为本次比较结果。

四组均从相同 FLUX.2、AE 和 ActionDiT 初始化开始，保持 `resume=null`、seed 42。不要拿其中一组训练后的策略权重初始化另一组消融。

| 实验 | 核心启动命令 | 独立输出目录示例 |
|---|---|---|
| baseline | `bash scripts/train.sh baseline` | `runs/formal_<tag>/baseline` |
| only pose | `bash scripts/train.sh only_pose` | `runs/formal_<tag>/only_pose` |
| only depth | `bash scripts/train.sh only_depth` | `runs/formal_<tag>/only_depth` |
| both | `bash scripts/train.sh both` | `runs/formal_<tag>/both` |

### 启用 W&B 实时监控

四组默认启用 W&B，并写入同一个 `imagewam_geometry` 项目。先用受托人的 W&B 账号登录，并将账号加入你能访问的共享团队／项目；GitHub 私有仓库权限不等于 W&B 权限。

```bash
source .venv/bin/activate
wandb login
# 替换成已授权的 W&B 团队名或用户名，不是 GitHub 用户名。
export WANDB_ENTITY="你的W&B团队或用户名"
export WANDB_MODE=online
```

下面的正式训练命令用同一个 `group` 汇总四组、用 `baseline / only_pose / only_depth / both` 区分 run。日志会打印 `W&B dashboard: ...`，受托人将项目链接和四个 run 链接交给你即可。W&B 页面可以实时叠加比较：

- `train/loss_action_raw`、`train/loss_video_raw`；有相应分支时查看 `train/loss_global_xyz`、`train/loss_relative_rotation`。
- `train/depth_abs_error_m`、`train/rotation_error_deg`、`train/rotation_endpoint_error_deg`。
- `train/lr`、`train/lr_group_*`、`train/grad_norm`、`performance/seconds_per_update` 和 `memory/peak_allocated_gib_max_rank`。

横轴为 `train/optimizer_step`；默认每 10 次更新上传一次训练指标，只有 rank 0 创建 run。超参数、全局 batch、几何分支开关和固定数据 revision 自动记录在 run config。速度指标是本次启动后累计平均值，包含前几步预热；精确测速使用上一节独立结果。不会自动上传训练数据、权重或视频。

可把 `WANDB_ENTITY` 和 `WANDB_MODE`（不是 API Key）写入 `.env.local`，让两个终端加载同一配置。服务器无法联网时显式设置 `export WANDB_MODE=offline`，下方启动函数会传给 `wandb.mode`；本地保存后可执行 `wandb sync /实际路径/wandb/offline-run-...`。离线期间网页不能实时更新，默认在线模式不会静默切成离线。参考 [W&B 官方说明](https://github.com/wandb/wandb)。

在线核验状态：已在网页创建并确认私有项目 [ivanhe191821-imagewam/imagewam_geometry](https://forge.coreweave.com/wandb/ivanhe191821-imagewam/imagewam_geometry)，但 L40 使用临时凭据请求 W&B API 返回 HTTP 403，尚未成功上传在线曲线；临时凭据已删除。SDK 离线记录和四卡训练写日志已验证。受托服务器要使用有该团队权限的账号，且能访问 W&B API；仅浏览器登录成功不代表服务器可以上传。若暂用离线模式，需要明确告知委托人。

### 按两对并行启动

需要同时有两组已分配且空闲的四卡资源（同一台八卡服务器，或两台各有四卡的服务器）。每个实验的四卡保持在同一节点；只有四张卡时无法执行上述配对并行计划。以下是非 Slurm 的八卡服务器示例，先确认 0–7 卡确实可用。

在两个终端中分别执行相同的准备命令；`RUN_TAG` 两边保持一致，并为本次正式实验选择未使用的名称：

```bash
set -euo pipefail
source .venv/bin/activate
source scripts/env.sh
export GPU_PER_NODE=4
export RUN_TAG=pair30k_20261118_01
: "${WANDB_ENTITY:?请先设置已授权的W&B团队或用户名}"
mkdir -p "runs/formal_$RUN_TAG"

run_one() {
  local mode="$1" gpu_ids="$2" port="$3"
  CUDA_VISIBLE_DEVICES="$gpu_ids" MASTER_PORT="$port" \
  bash scripts/train.sh "$mode" \
    batch_size=8 gradient_accumulation_steps=2 num_workers=8 \
    max_steps=40000 warmup_steps=1000 learning_rate=2.5e-5 \
    save_every=10000 log_every=10 seed=42 resume=null \
    wandb.enabled=true "wandb.mode=${WANDB_MODE:-online}" wandb.project=imagewam_geometry \
    "wandb.workspace=$WANDB_ENTITY" "wandb.group=$RUN_TAG" "wandb.name=$mode" \
    +record_metrics_every_step=true \
    "output_dir=./runs/formal_$RUN_TAG/$mode" \
    2>&1 | tee "runs/formal_$RUN_TAG/$mode.log"
}
```

第一对：终端 A 运行 `run_one baseline 0,1,2,3 29500`，终端 B 同时运行 `run_one only_depth 4,5,6,7 29501`。

确认第一对均达到 30k、30k 检查点完整保存，并停止和释放资源后，第二对：终端 A 运行 `run_one only_pose 0,1,2,3 29500`，终端 B 同时运行 `run_one both 4,5,6,7 29501`。终端因停止而退出时重新执行准备命令，保持同一个 `RUN_TAG`。每个实验目录只能用于对应的一次运行，不重复从第 0 步写入已有目录。

以上命令本身最多运行到 40k，不会自动在慢实验到 30k 时终止另一个实验；停止由受托人按本节规则执行。无需再做冒烟或短测。

Slurm 用户先申请同节点四卡和约 36 个 CPU 核心，再在作业内激活环境、`export GPU_PER_NODE=4` 并执行训练命令；保留调度器设置的可见卡。账号、分区、时间和内存限制按所在集群填写，本仓库不绑定 scut1 的账号。

如果 4×8 显存不足，可尝试四组统一改为 **4×4×累计4**，仍保持全局 batch 64。保持同一设置比较四组。不要随意跨节点；不要对正在训练的目录原地更新源码或数据。

## 10. 第八步：怎么检查训练结果、恢复训练

每组独立保存：

```text
runs/formal_<tag>/<mode>/
  config.yaml
  metrics.jsonl
  checkpoints/
    weights/step_010000.pt
    weights/step_020000.pt
    ...
    state/step_030000/            # 示例；默认只保留最近一次完整训练 state
```

检查示例（替换实际路径）：

```bash
tail -n 3 "runs/formal_$RUN_TAG/both/metrics.jsonl"
ls -lh "runs/formal_$RUN_TAG/both/checkpoints/weights"
```

关注 `step`、`loss_action_raw`、`loss_video_raw`。有旋转分支时看 `loss_relative_rotation`、`rotation_error_deg`、`rotation_endpoint_error_deg`；有深度分支时看 `loss_global_xyz`、`depth_abs_error_m`。辅助指标按分支出现；被关闭的辅助损失为零。不能因为各组总 loss 的项数不同，就直接比较总 loss 判断哪个更好。

最终四组比较应使用相同评测环境版本、任务、seed、回合数、观测／动作归一化、预测长度和执行长度，报告成功率及逐任务结果。本仓库没有打包模拟器场景和完整 rollout 服务，训练 loss 下降不能替代任务成功率。

恢复同一实验的完整 state：

```bash
bash scripts/train.sh both \
  resume=/absolute/path/to/checkpoints/state/step_020000 \
  output_dir=/absolute/path/to/the/same_or_new_run
```

`resume` 指向 **state 目录**才恢复优化器、scheduler 和步数；指向 `.pt` 文件只加载权重，会从新的优化器／学习率计划开始。恢复时保持实验类型、chunk、数据版本和必要训练配置一致。旧 chunk16 的 20k 权重有 16 个旋转 tokens，不能当作 chunk32 的严格完整断点；only pose / only depth / both 的辅助模块也不能互换严格恢复。

## 11. 推理：预测 32 步，执行前 16 步

训练仍监督 32 步，不因为部署只执行 16 步而屏蔽后半段。推理调用保持 `action_horizon=32`，执行端取前 16 个动作，然后重新获取当前 RGB/state 并预测下一组 32 步，丢弃上一轮剩余动作。

下面只是执行逻辑示意，`denormalize_actions`、`execute` 和相机/state 接口由部署端实现：

```python
prediction = policy.infer_action_flux2(
    prompt=instruction, input_image=current_rgb,
    proprio=current_state, action_horizon=32,
)
actions = denormalize_actions(prediction["action"], matching_training_stats)
execute(actions[:16])
# 重新取观测，再次调用策略；不要继续执行旧 chunk 的后 16 步。
```

输入 RGB/state 的预处理和动作反归一化必须与训练一致。几何头仍预测第 32 步的未来几何，不必改成第 16 步。相机 GT 位姿不作为部署输入。

## 12. 常见错误与检查边界

| 现象 | 处理 |
|---|---|
| `Dataset/cache version mismatch` | 检查是否仍指向旧 792 条目录或旧缓存；数据和缓存都切到当前固定 revision，不要关闭哈希检查。 |
| 789 条但视频/深度缺文件 | 按清单中路径补齐当前 revision；不要把旧编号文件复制进来顶替。 |
| 魔塔/HF 401 或 403 | 核对相应平台账号、模型条款与读取权限；GitHub 邀请不授予其他平台权限。 |
| 下载到新目录，训练仍读旧目录 | 编辑 `.env.local` 中两个数据变量，重新 `source scripts/env.sh`。 |
| ActionDiT 初始化缺失 | 下载 FLUX.2 主干后运行 `bash scripts/prepare_action_init.sh`。 |
| 端口被占用 | 为独立进程组指定不同 `MASTER_PORT`；不要终止其他人的进程。 |
| CUDA OOM | 先确认没有叠加其他训练；减小 microbatch 并等比例增加累积，记录实际设置；优化器状态导致的 OOM 不一定靠减 batch 能解决。 |
| 训练进程退出但只保存 step 0 | 检查 traceback，不能报告为训练更新成功。 |
| 只有 loss 没有成功率 | 当前是监督训练；成功率要在同一模拟器协议下单独 rollout。 |

已执行的验证包括新版元数据与缓存核对、789 条数据全部窗口枚举、八个任务首尾窗口解码、分支构建、真实 Depth Pro 解码器的 loss/梯度检查、窄版 25 层注意力／KV 缓存检查。完整 both 的 loss 和输入梯度已与 `bb16ef2` 旧实现核对一致。

L40 实测：四组均在 4 卡、每卡 bs1、梯度累计 1 下完成 **3 次优化器更新**，loss 有限且无 OOM；测试使用已有环境，未启用 CPU 优化器卸载。记录见 [VALIDATION.json](VALIDATION.json)。另做的 30 步速度统计见第 8 节；上述结果均不代表收敛。

README／默认路径复核见 [SETUP_AUDIT.json](SETUP_AUDIT.json)：在 L40 独立目录清除旧路径环境变量后，四组默认配置与启动命令、789 条轨迹的全部窗口和八个任务首尾样本、分支与 25 层 KV 检查均通过。空安装目标的依赖解析通过；独立安装 `modelscope-hub==0.7.1` 后，成功下载并核对四个固定版本元数据／缓存文件。复用了已有大体积数据和权重，没有重复下载整套数据或启动新的正式训练。

完整环境从零联网安装、四组 40k 收敛和模拟器成功率尚未验证；没有进行的检查不能由 Codex 推断为已通过。

## 13. 可以直接交给 Codex 的执行指令

```text
请在我指定的训练服务器上执行本仓库 README 的流程。
1. 先报告仓库分支和 commit、GPU 空闲情况、磁盘/配额与 Python/CUDA 环境。
   使用 main，不合并 pose-metric-pilot；不要覆盖本地未提交改动或已有正式训练。
2. 数据固定到 configs/dataset_release.json 中的 revision。
   同时准备 SIMPLE-RGBD-Teleop 和 SIMPLE-RGBD-Teleop-cache；已有副本先校验，
   不要因为数量接近就认定同版本，不要删除当前编号 37/398/425。
3. 按 README 安装环境、下载 FLUX.2 klein-base-4B、AE、Depth Pro，生成 ActionDiT 初始化。
   使用已有凭据登录；需要我交互登录时说明平台，不要在日志中输出 Token。
4. 对 baseline、only_pose、only_depth、both 做配置/权重检查；做全量窗口和首尾样本检查。
   预期789条、312139帧、311072个chunk32窗口，四组predict_translation均为false。
5. 代码提供方已完成结构检查、四组bs1冒烟和测速，不再重复短测。
   登录已授权的W&B账号，四组写入imagewam_geometry项目，同一group、四个独立run。
6. 环境和数据检查通过后，按global batch64、max_steps40000、warmup1000、
   lr2.5e-5、每10k保存、seed42启动，使用独立目录和端口；保持数据/源码固定。
   第一对baseline+only_depth，第二对only_pose+both；每个实验同节点四卡。
   每对并行运行，等两个实验都至少30k且30k权重完整保存后停止并释放资源。
   保留40k学习率计划，统一评测四组30k权重。脚本没有自动配对停止，需你负责监控。
   only_pose仅监督旋转；only_depth不创建旋转分支；both保持原loss权重。
7. 交付W&B项目及四组run链接，汇报配置、进度、指标和检查点路径。
   若训练失败先定位，不擅自改loss、冻结主干或更换优化器；不以训练loss代替成功率。
   部署预测32、执行16，需要在执行端重新取观测，不修改训练监督为16。
```

模型与第三方权重遵循各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)。不要将数据、环境、预训练权重、训练检查点或 `.env.local` 上传到 GitHub。
