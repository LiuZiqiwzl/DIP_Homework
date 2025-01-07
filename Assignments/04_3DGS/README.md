# Assignment 4 - Implement Simplified 3D Gaussian Splatting

## 作业成果展示

本次作业实现了一个完整的管道，用于从多视图图像重建由3DGS表示的3D场景。以下步骤使用 [chair 文件夹](data/chair) 中的数据；你也可以通过将图像放置在其他文件夹中来使用其他数据。

### 资源:
- [论文: 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [3DGS 官方实现](https://github.com/graphdeco-inria/gaussian-splatting)
- [Colmap for Structure-from-Motion](https://colmap.github.io/index.html)

### 步骤 0. 准备工作

1. 安装 Colmap:
   请参考 [Colmap 官方文档](https://colmap.github.io/install.html) 进行安装。

2. 设置环境变量:
   确保 Colmap 可执行文件所在路径已添加到系统的环境变量中（colmap.bat所在目录）。

3. 构建 PyTorch3D:
   必须保证cuda、cuda-toolkits、cudnn、pytorch版本完全匹配；
   在x64 Native Tools Command Prompt for VS 2019下完成包构建

### 步骤 1. 结构从运动恢复 (Structure-from-Motion)

首先，我们使用 Colmap 恢复相机姿态和一组3D点。
```
python mvs_with_colmap.py --data_dir data/chair
```

运行以下命令进行重建：
```
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

### 步骤 2. 简化的3D高斯点

#### 2.1 3D高斯初始化

计算3D协方差矩阵[此处的代码](gaussian_model.py#L115)。

#### 2.2 投影3D高斯以获得2D高斯

计算投影[此处的代码](gaussian_renderer.py#L49)。

#### 2.3 计算高斯值


计算2d高斯值[此处的代码](gaussian_renderer.py#L59)。

#### 2.4 体积渲染 (α-blending)

最终渲染计算 [此处的代码](gaussian_renderer.py#L83)。

实现后，在train同级目录打开终端输入：
```
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```
#### 3 结果展示

以下为训练5周期后结果：
<div style="display: flex; justify-content: center; align-items: center;"> <img src="image.png" alt="Result 1" style="width:300px;height:auto;"/> <img src="image-1.png" alt="Result 2" style="width:300px;height:auto;"/> <img src="image-2.png" alt="Result 3" style="width:300px;height:auto;"/> </div> ```