# ImageWAM Geometry on SIMPLE RGB-D

本仓库提供适配清洗后 SIMPLE RGB-D 数据的两个训练入口：`imagewam_baseline` 与 `imagewam_geometry`。基于 [ImageWAM](https://github.com/yuyangalin/ImageWAM)，不是上游官方发布。

| 配置 | 图像与动作预测 | 相对旋转预测 | 全局深度与 XYZ 监督 |
|---|---|---|---|
| imagewam_baseline | 有 | 无 | 无 |
| imagewam_geometry | 有 | 有 | 有 |

baseline 不创建姿态／几何 tokens 和对应预测头，不是仅将辅助 loss 权重设为零。两组共用数据读取、动作尾部掩码和原有主干处理方式。

## 当前实现

- action chunk 为 32，RGB 输入为 176×320；全局深度／XYZ 为 256×448，内参同步缩放。
- geometry 使用 FLUX.2 侧 32 个旋转 tokens、ActionDiT 侧 220 个全局几何 tokens，保留全部 25 层的注意力连接／条件 KV；取 5 个几何 token 中间状态交给多尺度适配层与 Depth Pro 解码器。
- 旋转监督覆盖 chunk 内有效未来帧；深度／XYZ 监督针对 chunk 末帧，XYZ 位于**未来相机坐标系**，单位为米。
- 当前没有局部 tokens，也不启用相对位移预测。相机位姿与深度只用于训练监督，不作为策略推理输入。
- 图像、动作、XYZ、旋转 loss 权重分别为 **0.5、1、0.5、5**。
- 保留真实尾部动作；不足 32 步的动作补齐部分不计 loss。没有真实 chunk 末帧时关闭该样本的未来图像与几何监督，旋转仅监督实际存在的未来帧。
- 当前图文时间调制恢复 baseline 行为；取消 geometry 额外的整层重计算，原有注意力与深度解码器的 checkpoint 策略仍保留。

## 环境

目标平台为 Linux + NVIDIA GPU。已用 Python 3.11、PyTorch 2.7.1/CUDA 11.8、Transformers 4.56.1、Accelerate 1.12.0 和 DeepSpeed 0.18.5 运行。

先接受本 GitHub 私有仓库的协作邀请，并配置自己的 GitHub Git 登录。以下命令均在仓库根目录执行；已克隆的用户直接进入现有目录即可。需要预先安装 Git、Python 3.11 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/Ivan-hexy/imagewam_geometry.git
cd imagewam_geometry
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[shared]'
cp -n .env.example .env.local
```

FLUX.2 和 Depth Pro 的必要源码快照已放在 `third_party/`。不需要再安装它们的完整 pyproject 依赖（会改变当前 PyTorch 版本）。GPU 驱动和需要编译 CUDA 扩展时使用的 CUDA toolkit 由训练机器提供。

## 下载数据

### 1. 获得权限并登录魔塔

数据与文本缓存位于 [ivanhe666/simple_rgbd 的文件页](https://modelscope.cn/datasets/ivanhe666/simple_rgbd/files)。**数据仓库在打包时为私有：GitHub 代码访问权限不等于魔塔数据访问权限。** 合作者需要向数据所有者申请魔塔读取权限，然后使用自己的魔塔账号登录。CLI 提示时输入该账号的魔塔访问 Token；不要使用 GitHub 或 Hugging Face Token，也不要将 Token 写进仓库。

下载工具使用独立环境，无需激活它，不会替换训练环境的依赖：

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install modelscope-hub==0.7.1
.download-venv/bin/ms-hub login
```

### 2. 下载固定版本的数据和缓存

```bash
bash scripts/download_data.sh
```

该脚本等价于下面的命令，**两种方式选一种即可**：

```bash
.download-venv/bin/ms-hub download ivanhe666/simple_rgbd \
  --repo-type dataset \
  --revision a36b289eb6d6d7c6b36e8ee11f7c8a03ac3730f6 \
  --local-dir ./data \
  --max-workers 4
```

固定版本包含 **792 条轨迹、313,553 帧、8 个任务**，含尾部处理后有效动作采样窗口为 **312,481**。缓存内旧的 full-window 计数不包含补齐尾部窗口。数据和缓存约需 **200 GB**，请另留模型权重、环境及训练检查点空间。不要只下载 RGB 视频；动作、深度、相机参数、清单、归一化统计和文本缓存都需要保留。

默认下载后的目录如下，两个 `SIMPLE-RGBD-*` 目录必须并列：

```text
imagewam_geometry/
  data/
    SIMPLE-RGBD-Teleop/
      meta/
        episodes.jsonl
      ...                         # RGB、深度、动作和轨迹等文件
    SIMPLE-RGBD-Teleop-cache/
      all_verified_stats.json
      all_verified_training_manifest.json
      qwen3/                      # 训练使用的预计算文本特征
```

### 3. 可选：放到独立数据盘或使用已有数据

例如下载到 `/mnt/datasets/simple_rgbd`（替换成自己的可写目录）：

```bash
DATA_DOWNLOAD_DIR=/mnt/datasets/simple_rgbd bash scripts/download_data.sh
```

然后**编辑**仓库根目录 `.env.local`，设置以下两行；已有同版本副本也直接这样配置，无需重复下载：

```bash
export SIMPLE_RGBD_ROOT=/mnt/datasets/simple_rgbd/SIMPLE-RGBD-Teleop
export SIMPLE_RGBD_CACHE=/mnt/datasets/simple_rgbd/SIMPLE-RGBD-Teleop-cache
```

`DATA_DOWNLOAD_DIR` 只控制下载位置，不会自动修改训练路径。训练过程中不要重命名、重编号或替换这一版本的数据；更换数据需使用新路径并重新构造加载器。

## 下载模型与生成 ActionDiT 初始化

### 1. 需要哪些文件

GitHub 只保存代码，预训练权重从以下官方来源下载。表中路径均相对于本仓库根目录。

| 内容 | 来源与准确文件名 | 默认保存位置 | 哪组需要 |
|---|---|---|---|
| FLUX.2 图像主干 | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/main) 中的 `flux-2-klein-base-4b.safetensors` | `checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors` | 两组 |
| FLUX.2 图像自编码器 AE | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main) 中的 `ae.safetensors` | `checkpoints/flux2/FLUX.2-dev/ae.safetensors` | 两组 |
| Depth Pro 几何解码器初始化 | [Apple 官方权重](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt)，见 [官方下载脚本](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) | `checkpoints/depth_pro/depth_pro.pt` | 仅 geometry |
| ActionDiT 初始化 | 由本仓库脚本根据上述 FLUX.2 主干生成，无需另找下载链接 | `checkpoints/action_dit_flux2_4b_simple9_36d_init.pt` | 两组 |

必须使用 **klein-base-4B**，不要替换成蒸馏版 `FLUX.2-klein-4B`、9B 或完整 `FLUX.2-dev` 主干。从 `FLUX.2-dev` 仓库只下载 `ae.safetensors`，不需要下载整个模型仓库。

### 2. 登录 Hugging Face

使用自己的 Hugging Face 账号访问上表中的两个 FLUX.2 仓库；若页面要求申请访问或接受模型条款，先在网页完成。然后在 [Hugging Face Token 设置](https://huggingface.co/settings/tokens) 创建有权读取相应模型的 Token，在训练环境执行：

```bash
source .venv/bin/activate
hf auth login
hf auth whoami
```

`hf auth login` 会交互式请求 Token。这里只做文件下载，询问是否加入 Git 凭据时可选 `n`。本项目固定的 `huggingface-hub==0.34.4` 支持上述命令，详见 [对应版本 CLI 文档](https://huggingface.co/docs/huggingface_hub/v0.34.4/guides/cli)。这是 Hugging Face 登录，与 GitHub、魔塔的登录相互独立。

### 3. 下载预训练权重

运行 geometry，或者准备同时运行两组实验：

```bash
source scripts/env.sh
python scripts/download_weights.py --mode geometry
```

如果**只运行 baseline**，将上一条 Python 命令换成：

```bash
python scripts/download_weights.py --mode baseline
```

`geometry` 下载 FLUX.2 主干、AE 和 Depth Pro；`baseline` 只下载前两项。两组共用同一份 FLUX.2 和 ActionDiT 初始化，不需要各下载一套。下载脚本打印每个文件的本地路径。

若希望手动下载，下面是与默认目录一致的命令；使用过上面的脚本就不必重复执行：

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B \
  flux-2-klein-base-4b.safetensors \
  --local-dir ./checkpoints/flux2/FLUX.2-klein-base-4B
hf download black-forest-labs/FLUX.2-dev ae.safetensors \
  --local-dir ./checkpoints/flux2/FLUX.2-dev

# 仅 geometry 需要；使用临时文件，下载成功后再改名
mkdir -p ./checkpoints/depth_pro
curl --fail --location --retry 3 \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  --output ./checkpoints/depth_pro/depth_pro.pt.download && \
  mv ./checkpoints/depth_pro/depth_pro.pt.download ./checkpoints/depth_pro/depth_pro.pt
```

### 4. 生成 ActionDiT 初始化

下载完 FLUX.2 主干后，在已激活的训练环境执行一次：

```bash
bash scripts/prepare_action_init.sh
```

该脚本默认在 **CPU** 上转换权重，并写入 `checkpoints/action_dit_flux2_4b_simple9_36d_init.pt`；转换期间会占用主机内存。文件已存在时会跳过。文件名中的 `simple9` 是保留的历史命名，不代表当前数据有 9 个任务。ActionDiT 主干由 FLUX.2 权重初始化，36 维动作输入输出投影在训练时初始化。

完整 geometry 准备结果为：

```text
checkpoints/
  flux2/
    FLUX.2-klein-base-4B/
      flux-2-klein-base-4b.safetensors
    FLUX.2-dev/
      ae.safetensors
  depth_pro/
    depth_pro.pt
  action_dit_flux2_4b_simple9_36d_init.pt
```

### 5. 可选：使用其他位置的已有权重

在 `.env.local` 编辑所需路径，再执行 `source scripts/env.sh`：

```bash
export FLUX2_MODEL_PATH=/mnt/models/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/mnt/models/ae.safetensors
export SIMPLE_ACTION_INIT=/mnt/models/action_dit_flux2_4b_simple9_36d_init.pt
export DEPTH_PRO_CHECKPOINT=/mnt/models/depth_pro.pt
```

只设置自己需要覆盖的项。已有文件可跳过对应下载；ActionDiT 尚未生成时仍需运行初始化脚本。**当前自动下载脚本中的两个 FLUX.2 文件始终写入仓库默认目录**，`FLUX2_MODEL_PATH` / `FLUX2_AE_MODEL_PATH` 控制加载位置，不会改变下载目录；`DEPTH_PRO_CHECKPOINT` 则同时控制 Depth Pro 的下载和加载位置。不要将新下载的权重和 `.env.local` 指向的旧文件混淆。

### 6. 下载完成后检查

在仓库根目录执行；只跑一组时只执行对应检查即可：

```bash
source .venv/bin/activate
source scripts/env.sh
python scripts/check_setup.py --mode baseline --require-weights
python scripts/check_setup.py --mode geometry --require-weights
```

成功时输出 JSON，其中 `config_passed` 和 `sampled_first_and_last_windows_passed` 均为 `true`，并显示 `episodes: 792`、`frames: 313553`、`sampled_tasks: 8`。该检查核对权重文件存在性、固定版本的 episode 清单哈希、所有 episode 文件是否齐全，以及每个任务的首尾采样窗口；**不会完整加载验证模型权重，也不是对约 200 GB 数据逐文件重算哈希**。

训练复用数据包 `SIMPLE-RGBD-Teleop-cache/qwen3/` 的文本特征，**上述训练流程无需下载 Qwen3 模型**。在线处理新指令需要另行准备文本编码器。本仓库的下载流程用于从官方预训练权重开始训练，**不包含之前训练到 30k／40k 的策略检查点，也不是评测权重下载入口**。各权重分别遵循其原始许可。

### 常见下载问题

| 现象 | 处理方式 |
|---|---|
| 魔塔返回 401/403，或已知私有仓库显示不存在 | 先确认自己的魔塔账号已获得数据读取权限，再用 `.download-venv/bin/ms-hub login` 登录；GitHub 邀请不会授予魔塔权限。 |
| Hugging Face 返回 401/403 / `GatedRepoError` | 用 `hf auth whoami` 确认账号，检查网页访问申请与 Token 读取权限；尤其注意 AE 来自独立的 `FLUX.2-dev` 仓库。 |
| Hugging Face 下载超时 | 检查训练服务器能否访问 Hugging Face；可执行 `export HF_HUB_DOWNLOAD_TIMEOUT=60` 后重试同一命令。 |
| 文件已下载，但检查仍报找不到 | 重新 `source scripts/env.sh`，核对 `.env.local`；数据路径应分别指向数据和缓存子目录，权重路径应指向具体文件。 |
| 报 `Wrong dataset version` 或 manifest 不一致 | 确认下载了指定提交的数据和配套缓存，且路径没有指向旧数据；不要通过关闭版本检查绕过。 |
| 找不到 ActionDiT 初始化文件 | 确认 FLUX.2 主干下载完成，然后执行 `bash scripts/prepare_action_init.sh`。 |

## 启动训练

先运行配置检查或查看将执行的命令：

```bash
python scripts/check_setup.py --mode baseline --config-only
python scripts/check_setup.py --mode geometry --config-only
DRY_RUN=true bash scripts/train.sh geometry
```

两个实验分别启动，不要让以下两个进程占用同一组 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_PER_NODE=4 bash scripts/train.sh baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_PER_NODE=4 bash scripts/train.sh geometry
```

默认：每卡 batch 8、梯度累积 2、8 个 workers，全局 batch 为 64；40,000 次更新，学习率 2.5e-5、1000 步预热、余弦下降，每 10,000 步保存一次。使用 ZeRO-1，优化器与参数均不卸载到 CPU。输出位于各自的 `runs/` 子目录。Slurm 用户应先在**同一个节点**申请四卡及约 36 个 CPU 核心，再运行上述命令；此处没有绑定任何特定集群账号或分区。

独立 200 步冒烟测试（其学习率计划与正式训练不同）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_PER_NODE=4 bash scripts/train.sh geometry   max_steps=200 warmup_steps=10 save_every=0 eval_every=0   +checkpoint_mode=none +record_metrics_every_step=true
```

在单节点四张 A800 80GB 上，当前 geometry 的 4×8、累积2配置已完成200次更新，稳定约3.10秒/次完整更新。4×16无累积在该配置下OOM。速度和显存会随硬件与环境变化；不要把这次短测当成任务成功率验证。旧 chunk16 的几何检查点包含16个姿态tokens，不能直接作为 chunk32 的严格完整断点恢复。

## 代码来源与共享范围

核心模型源码保持打包时已经测试的实现，包装层只增加可移植配置、下载、检查与启动入口。完整环境的从零安装和模拟器成功率评测尚未在这个导出目录重新运行。保留了上游核心模块，当前共享入口只支持上述 FLUX.2／SIMPLE RGB-D 训练流程；不包含模拟器场景资产和完整评测服务。

参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)。`only pose`、`only depth` 尚未作为真正独立网络分支消融实现，不能通过单纯置零 loss 冒充这两组实验。

## 上传到 GitHub

在 `Ivan-hexy` 账号创建一个空仓库，例如 `imagewam_geometry`。若只给合作者共享可先设为 Private；要公开时再选择 Public。创建空仓库时不要额外生成 README、LICENSE 或 .gitignore，这些文件已包含在本目录。

```bash
git init
git add .
git status --short
git commit -m "Share SIMPLE RGB-D ImageWAM baseline and geometry"
git branch -M main
git remote add origin https://github.com/Ivan-hexy/imagewam_geometry.git
git push -u origin main
```

使用自己已有的 GitHub 登录／凭证管理器或 SSH 密钥完成认证；不要将密码或 Token 写进远程 URL。不要把本目录之外的服务器文件夹一起上传。`.gitignore` 已排除数据、环境、权重、日志和本地配置。
