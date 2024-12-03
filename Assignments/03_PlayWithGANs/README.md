### Resources:
- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN) & [Implementaion 2](https://github.com/OpenGVLab/DragGAN)
- [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment)

---

\
See [作业03-Play_with_GANs.pptx](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49) for detailed requirements.

## DIP 作业 3 Play with GANs



## 功能描述

 Part 1: Increment hw2 with Discriminative Loss 

    见作业2部分

 Part 2: Combine DragGAN with Automatic Face Landmarks

    将DragGAN与面部特征点（68个）自动标记结合，实现对DragGan内的模型ai生成人物面部图，自动变形。

## 问题分析

 Part 1: Increment hw2 with Discriminative Loss 

    见作业2部分

 Part 2: Combine DragGAN with Automatic Face Landmarks
 
    根据面部特征点对应索引，采用传统方法移动坐标

## 前置要求

 Part 2：
1. git clone https://github.com/XingangPan/DragGAN.git
2. conda install -c 1adrianb face_alignment
注：setuptools包不能高于72.0.0版本，gradio不能高于3.8.1版本。pillow必须低于10版本
3. 下载我的github.com/LiuZiqiwzl/DIP_Homework.git中Assignments\03_PlayWithGANs\visualizer_draggan.py文件，放入DragGAN目录下覆盖。
4. cd DragGAN，在DragGAN目录下运行：python scripts/download_model.py

## 运行程序


进入目录：
cd DragGAN

运行：
.\scripts\gui.bat

选择图片后，点击apply landmarks获取特征点；点击expression选择表情，再点击apply expression应用表情改变目标点。
最后点击start
![alt text](image.png)

## 结果展示


### Part2

| ![alt text](效果图/00005.png) | ![alt text](效果图/smile.png) |
| --- | --- |
| ![alt text](效果图/00006.png) | ![alt text](效果图/close_eyes.png) |

![alt text](closeeyes_1.gif)

![alt text](draggan_1.gif)

## 代码出处

https://github.com/XingangPan/DragGAN
https://github.com/1adrianb/face-alignment

