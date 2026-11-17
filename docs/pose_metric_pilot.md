# 相对位姿预测：实验版本

本分支基于 `bb16ef2` 的完整 geometry 模型，在相同 32 个 pose tokens 上
启用 3D 相对平移头。6D 旋转表示、220 个几何 tokens、全部 25 层注意力、
ActionDiT、Depth Pro 与尾部掩码保持不变。原来的 `imagewam_geometry`
配置仍默认只监督旋转；新入口为 `imagewam_geometry_pose`。

**状态：CPU 数学与掩码检查通过；单卡前后向和第 0 步评估通过，但首个
AdamW foreach 更新显存不足。改用逐张量 AdamW 的 200 步重试正在排队，
尚无收敛结论，不是已经验证优于旧 loss 的正式推荐配置。**

## 损失

设 θ 为预测与真实旋转的 SO(3) 测地角（弧度），d 为平移向量的欧氏误差
（米），则：

```
L_pose = 0.5 × [mean(θ / s_R) + mean(d / s_t)]
L_total = 0.5 L_image + L_action + 0.5 L_XYZ + 0.1 L_pose
s_R = 0.021099236770746314 rad ≈ 1.2089°
s_t = 0.01463603222844109 m ≈ 1.4636 cm
```

先对每个窗口的有效时刻平均，再对有效窗口平均。没有未来观测的窗口
不监督位姿；真实尾部动作仍参与动作监督。所有未来位姿均相对当前相机，
平移采用当前相机 ROS 坐标系，深度 XYZ 仍在未来相机坐标系监督。

这两个尺度取自 64 条训练轨迹、512 个窗口的静止预测平均误差的中位数，
训练中固定不变。留出轨迹不参与拟合。它们不是每个样本自身的位移/角度，
所以静止片段不会出现除零。外部权重 0.1 是初始实验设置，并非最佳值；
新目标的数值尺度变了，不能直接和旧权重 5 比大小。

旋转使用 FP32 atan2 实现，避免 clamp-acos 在小角度处产生人工梯度死区。
距离在零处使用 PyTorch 的零次梯度；测地距离在 180° 处本身不可微。
旧平方弦距离也有数学依据，本实验仅检验直接优化物理误差是否更合适。

## 参考与改动边界

- [6DRepNet](https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/loss.py)：旋转测地距离。
- [Kendall & Cipolla, CVPR 2017](https://arxiv.org/abs/1704.00390)：位姿回归的平移/旋转平衡问题。
- [VGGT 官方训练实现](https://github.com/facebookresearch/vggt/blob/main/training/loss.py)：使用 L1 相机残差，其作者报告在该模型中更稳定。

固定数据尺度归一化是本项目的组合设计，不是对上述论文公式的原样复现。
本次没有引入可学习不确定性权重，避免短测中通过权重下降掩盖预测误差；
也没有引入依赖对应点和遮挡处理的重投影损失。

## 复现 200 次更新

使用原 README 的环境和预训练权重。数据支持原始 792 条版本，以及删除
三条问题垃圾桶轨迹后的 789 条版本；本次实际测试使用后者。魔塔下载脚本
仍固定到原始 792 条提交，不会自动下载或生成 789 条版本。

```bash
source .venv/bin/activate
source scripts/env.sh
python scripts/verify_metric_pose_loss.py
python scripts/prepare_pose_pilot.py
python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 1 scripts/train.py task=pose_geodesic200 \
  output_dir=./runs/pose_geodesic200
```

准备脚本按源轨迹留出每任务一条，生成分割清单、固定尺度及实验配置。
每个任务分别从训练/留出轨迹取两个固定窗口，在 0、100、200 步使用固定
噪声评估；记录平均/中位/p90 误差、chunk 末端误差和静止预测参照。
沿用已有动作归一化缓存，因此这是工程短测，不是重新拟合全部统计量的
严格独立基准。没有机器人 rollout，不代表任务成功率。

单卡 microbatch 1、累计 16（总 batch 16），200 次更新，峰值学习率
2.5e-5，warmup 10 步后 cosine。`optimizer_foreach: false` 只改变 AdamW
张量更新方式，以减少临时显存；没有 CPU 卸载、量化优化器或冻结主干。
不保存大模型检查点，只保存指标和固定评估记录。

继续正常训练可使用 `task=imagewam_geometry_pose`。若更换数据，应重新拟合
尺度，而不是将本次厘米/角度尺度当作通用常数。
