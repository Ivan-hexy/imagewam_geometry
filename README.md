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

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[shared]'
cp .env.example .env.local
```

FLUX.2 和 Depth Pro 的必要源码快照已放在 `third_party/`。不需要再安装它们的完整 pyproject 依赖（会改变当前 PyTorch 版本）。GPU 驱动和需要编译 CUDA 扩展时使用的 CUDA toolkit 由训练机器提供。

## 下载数据

数据与文本缓存托管于 [ivanhe666/simple_rgbd](https://modelscope.cn/datasets/ivanhe666/simple_rgbd)。固定使用提交 `a36b289eb6d6d7c6b36e8ee11f7c8a03ac3730f6`：**792 条、313,553 帧、8 个任务**。含尾部处理后有效动作采样窗口为 312,481；缓存内旧的 full-window 计数不包含补齐尾部窗口。

数据仓库在打包时是私有的；合作者需要先获得该仓库的读取权限，并在自己的机器登录。不要在代码中写入 Token。下载工具使用独立环境：

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install modelscope-hub==0.7.1
.download-venv/bin/ms-hub login
bash scripts/download_data.sh
```

约需 200 GB，下载结果为两个并列目录：

```text
data/
  SIMPLE-RGBD-Teleop/
  SIMPLE-RGBD-Teleop-cache/
    all_verified_stats.json
    all_verified_training_manifest.json
    qwen3/
```

也可在 `.env.local` 中设置 `SIMPLE_RGBD_ROOT` 和 `SIMPLE_RGBD_CACHE` 指向已有副本。训练过程中不要重命名、重编号或替换这一版本的数据。更换数据需使用新路径并重新构造加载器。

```bash
source scripts/env.sh
python scripts/check_setup.py --mode geometry
```

此检查核对固定版本清单、完整文件存在性及每个任务首尾采样窗口；不是重新计算全部 200 GB 文件的哈希。

## 下载模型与生成 ActionDiT 初始化

根据官方模型仓库要求，先在自己的 Hugging Face 账号获得 FLUX.2 模型访问权限，再登录：

```bash
huggingface-cli login
python scripts/download_weights.py --mode geometry
bash scripts/prepare_action_init.sh
```

只运行 baseline 时可用 `--mode baseline` 跳过 Depth Pro 权重。ActionDiT 初始化由 FLUX.2 官方图像权重生成，36 维动作输入输出投影仍在训练时初始化。训练复用数据包中的 Qwen3 文本缓存，因此不需要另外下载 Qwen3 模型；在线处理新指令需要另行准备文本编码器。

大权重与训练结果不属于代码仓库，分别遵循原模型许可。当前仓库不包含之前训练到 30k／40k 的策略权重。

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
