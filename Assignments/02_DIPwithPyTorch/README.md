## DIP 作业 2：Poisson 图像编辑

## 简介
1. 泊松图像编辑
   使用opencv库实现多边形蒙版计算
   laplace_loss 计算

    ![alt text](image-1.png)

    ![alt text](image-2.png)

通过loss函数：用拉普拉斯卷积核[[0, 1, 0], [1, -4, 1], [0, 1, 0]]求出各点散度，并计算与前景图像各点散度插值方差和，
，迭代优化laplace方程得出近似解。

2. pix2pix
   添加一些卷积层和对应的反卷积层，forwad函数中完成调用。

## Requirements

    conda install -c conda-forge opencv
    conda install pytorch 12.4
    bash download_facades_dataset.sh

## Running

    1.泊松图像编辑
    run: run_blending_gradio.py 

    2.pix2pix
    run:在pixtopix文件夹处运行train.py  

## Results

1.泊松图像编辑
<!-- 第一行 -->
<div style="display: flex; align-items: center;">
    <img src="image-3.png" alt="Image 3" style="width:300px;height:auto;"/>
    <img src="image-5.png" alt="Image 5" style="width:300px;height:auto;"/>
</div>

<!-- 第二行 -->
<div style="display: flex; align-items: center;">
    <img src="image-4.png" alt="Image 4" style="width:300px;height:auto;"/>
    <img src="image-6.png" alt="Image 6" style="width:300px;height:auto;"/>
</div>

2.pix2pix

<!-- 前两个图像各自单独一行 -->
<div style="text-align: center;">
    <img src="Pix2Pix/val_results/epoch_795/result_1.png" alt="Result 1" style="width:300px;height:auto;"/>
</div>
<div style="text-align: center;">
    <img src="Pix2Pix/val_results/epoch_795/result_3.png" alt="Result 3" style="width:300px;height:auto;"/>
</div>

<!-- 后两个图像在同一行显示 -->
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="Pix2Pix/doro.jpg" alt="Doro" style="width:350px;height:auto;"/>
    <img src="Pix2Pix/output_doro.png" alt="Output Doro" style="width:350px;height:auto;"/>
</div>
