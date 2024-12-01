### Resources:
- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN) & [Implementaion 2](https://github.com/OpenGVLab/DragGAN)
- [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment)

---

\
See [作业03-Play_with_GANs.pptx](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49) for detailed requirements.

## DIP 作业 3 Play with GANs

## Part 1: Increment hw2 with Discriminative Loss 

见作业2部分

## Part 2: Combine DragGAN with Automatic Face Landmarks

## 功能描述

本项目实现了图像的基本变换功能，包括：

图像的旋转、平移、缩放和镜像处理。
基于点的 MLS（Moving Least Squares）变形。
MLS 变形
MLS 变形通过 mls_affine 函数实现，该函数用于对单个点进行变形计算。

## 问题分析

为了提高 MLS 变形的效率，我们采用了以下方法：
将显示for循环转化为向量运算
使用 joblib 实现并行运算
将 NumPy 数组转换为 CuPy（cp）数组，以利用 GPU 加速
然而，上述优化效果并不理想，分析原因如下，

mls_affine 函数主要是对单个点进行变形后坐标的计算，其内部运算量有限，通常不超过百次。因此，无论是并行计算还是 GPU 加速，提升效率的空间都比较有限，时间主要浪费在对函数的调用、数据类型的转换和传递。NumPy 类型数组本身就支持并行运算，因此并行处理带来的效率提升并不明显；而转化为 CuPy 数组后，反而增加了处理时间。

改进方案
目前采用文章中提到的网格分割图像的方法，对网格顶点进行转换。随后，针对每个网格内部，计算其对应的整体仿射矩阵，并批量进行转换。然而，由于实施的是正向映射（即计算原图像像素点在变形后图像中的新坐标），这会导致变形后图像的一些点未被映射，从而出现黑线现象。

由于时间限制，未能采用反向映射的方法来解决黑线问题，而是使用了后期处理的方法，包括中值滤波、双线性插值和三次插值等，虽然如此，仍然无法完全避免黑线现象的发生。

未来的优化方向
GPU 加速方案的失败可以通过直接修改函数实现来解决，即将整个图像的像素点转为向量，并同时计算其变形后的位置。不过由于时间原因，未能在当前版本中实现这一思路。

## 前置要求

git clone github.com/LiuZiqiwzl/DIP_Homework.git
python -m pip install -r requirements.txt
注：本次作业使用 conda（conda-forge） 安装的 gradio 包版本为 3.46.1，版本过高有 bug（gr.Image 不支持参数 source, 且不支持点选取）

## 运行程序


进入目录：
cd Assignments/01_ImageWarping/


根据需要，运行以下命令：

- **若要执行基本变换，请运行：**
python run_global_transform.py


- **若要执行点引导变换，请运行：**
python run_point_transform.py



## 结果展示

### Basic Transformation

<img src="pics/global_transform.gif" alt="alt text" width="600">

<img src="image.png" alt="alt text" width="700">

### Point Guided Deformation:

<img src="pics/point_transform_v2.gif" alt="alt text" width="800">
<img src="pics/image.png" alt="变形后效果" width="800">

更多效果图在 pics 文件夹内

蓝色为控制点 青色为目标点，绿色是变形前网格，红色是变形后网格。
关闭网格显示，可删除 run_point_transform.py 网格绘制之后的内容。

## 理论基础

> 📋 参考资料 [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).


part1: 改进第二次作业 完成