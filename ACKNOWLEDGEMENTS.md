# Acknowledgements

本课程项目的完成离不开学术界和开源社区诸多极具启发性的工作。在此我们诚挚地声明由于代码高度派生，本工程参考、复用或继承了以下优秀开源项目的骨架、代码与思想：

## 1. 核心底层 Backbone

**[2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting)**  
*Authors: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao.*

这是本项目的**唯一绝对底层与算法基座**。
- 我们直接继承了其在场景表示（surfel representations）、光度及几何畸变损失上的基本实现。
- 我们直接内置拷贝了其原本的 `scene/`, `gaussian_renderer/`, 和有关相机/参数系统的环境基建。
- 我们使用了其附属的 `diff-surfel-rasterization` 纯可导软栅格化 CUDA 扩展项目环境。

## 2. 局部正则项设计思想来源

**[SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://github.com/Anttwo/SuGaR)**  
*Authors: Antoine Guédon, Vincent Lepetit.*

本课程项目中的核心代码修改——`alignment_loss.py` 中关于点到局部切面的距离衰减机制 ($L_{plane}$) 及局部法向相似度约束 ($L_{normal}$)——主要是由于吸纳与致敬了 SuGaR 所强调的 “Mesh-friendly” 和 “Surface-aligned” 思路。

虽然我们**没有**在代码中并入 SuGaR 庞杂的数据与 SDF 提取组件，但其正则函数的公式设计在提升我们实验里 2DGS 的几何一致性环节起到了关键的思想指导。

## 3. 启蒙平台与第三方依赖

**[3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting)**  
*Authors: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis.*

2DGS 及本仓库所共用的上层 `train.py` 控制流、点云初始化、SH球谐计算、`plyfile` IO解析以及基于 `simple-knn` 的邻域计算思路均发源于 3DGS。它为当前的显式渲染赛道打下了稳固的接口基础。
