# 基于表面导向 Gaussian Splatting 的几何一致与网格友好三维重建

**3D-mapping** 是一个三维重建课程项目，核心建立在 [2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting) 的主骨架之上。

本项目通过在原 2DGS 的光度与几何损失基础之上，引入受 [SuGaR](https://github.com/Anttwo/SuGaR) 启发的**局部表面对准正则化项（Alignment Regularization）**，以在保持高保真新视角合成的同时，提高网格友好的局部几何一致性。

## 主要特性

- **唯一 Backbone**：原生保留了完整可运行的 2DGS 渲染库、场景表达和优化进程，保证最先进的前向渲染表现。
- **Alignment 正则**：
  - **L_plane**: 惩罚 3D KNN 局部邻域内的高斯点偏离其中心切平面的距离。
  - **L_normal**: 提升局部邻域内高斯基元法向向量的一致性。
- **可复用的架构**：通过 `configs/default.yaml` 可无缝切换 `baseline` (原始 2DGS 逻辑) 与 `ours` (激活对齐正则) 模式，方便设计消融对照实验。
- **显存保护机制**：独立重新设计的 Chunk-based Graph 采样器，保障百万级别 splats 进行高强度 KNN 查询时不发生 OOM（Out Of Memory）。

## 安装

本项目作为一个**独立派生仓库**，已经将所需的第三方依赖代码（CUDA 扩展）内置于 `third_party` 目录下，您无需再执行任何 submodule 拉取命令。

请准备一个支持 CUDA 的 Python 3.10+ 环境（示例使用 Conda）：
```bash
conda create -n 3d-mapping python=3.10
conda activate 3d-mapping
# 根据您的系统和 CUDA 版本，安装适配的 PyTorch (这里以 CUDA 11.8 为例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**关键步骤：安装内置的第三方本地扩展：**
注意：请确保系统中包含可用的 CUDA 编译器 (`nvcc`)。
```bash
pip install -e third_party/simple-knn
pip install -e third_party/diff-surfel-rasterization
```

安装其他常规依赖库：
```bash
pip install pyyaml plyfile tqdm scipy
```

## 运行

程序的运行入口点是 `train.py`。你可以通过附加 `--config` 开关来决定策略，同时也需遵守 2DGS 常规的路径参数。

### 跑带有 Alignment Loss 的前沿模式 (Ours)
```bash
python train.py -s <path_to_COLMAP_or_NeRF_synthetic_dataset> --config configs/default.yaml
```
*(在 `default.yaml` 中，`mode` 应该被设为 `ours`)*

### 跑原始 2DGS (Baseline)
你只需要在 `configs/default.yaml` 中将 `mode` 设为 `baseline`。这会强制让 `lambda2=0`，完全退回到原始 2DGS 状态。
```bash
python train.py -s <path_to_COLMAP_or_NeRF_synthetic_dataset> --config configs/default.yaml
```

## 测试与快速验证

我们推荐使用轻量的 Blender (NeRF Synthetic) 的 Lego 数据集通过少量迭代进行快速的训练链路验证。

### 1. 快速验证原始 2DGS 逻辑 (Baseline 模式)
在 `configs/default.yaml` 中将 `mode` 设为 `baseline`，然后运行 100 iterations：
```bash
python train.py -s data/nerf_synthetic/lego -m output/baseline_test --iterations 100 --config configs/default.yaml
```

### 2. 快速验证带有 Alignment Loss 的模式 (Ours)
在 `configs/default.yaml` 中将 `mode` 设为 `ours`，测试正则算法前反向传播：
```bash
python train.py -s data/nerf_synthetic/lego -m output/ours_test --iterations 100 --config configs/default.yaml
```

## 参数说明 (`configs/default.yaml`)

```yaml
mode: ours # 或 'baseline'
align:
  enable: true           # 总开关
  beta1: 1.0             # L_plane 项在 L_align 内部的权重
  beta2: 0.1             # L_normal 项在 L_align 内部的权重
  knn_k: 10              # KNN 半径点数
  knn_chunk_size: 4096   # OOM保护：批次查 KNN 的分块数
  sigma_d: 0.01          # 距离指数衰减惩罚权重
  lambda2_max: 0.5       # 全局最终的 Alignment Loss 最大权重
  warmup_iters: 3000     # 训练迭代多少次后才渐渐开始引入 Alignment Loss
  ramp_iters: 4000       # 从 0 爬伸到 lambda2_max 需花费的步数
```

## 数据集结构

期望的数据集格式为标准 COLMAP 处理后或 NeRF-Synthetic 原生形式，要求与 3DGS/2DGS 的组织一致。

数据集下载

[NeRF Synthetic 的 lego 场景](https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/tree/main/lego)

```powershell
python download_lego.py
```

## 项目实现进程说明

本项目现处用于实现课程目标和评测的**第一阶段（最小骨架阶段）**，包含的独立模块位于：
- `losses/alignment_loss.py`: 完全可导的张量正则组件。
- `utils/knn_graph.py`: Chunk-based 的动态邻域生成器。
- `utils/schedules.py`: 针对学习进程的标量坡度调度器。