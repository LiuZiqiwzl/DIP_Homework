import cv2
import numpy as np
import gradio as gr
import numpy as np
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import pairwise_distances
from MLS import *
import time

from skimage.draw import polygon
  


# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image
 

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """

    source_pts = np.array(source_pts)
    target_pts = np.array(target_pts)

    # 点列异常条件判断
    if len(source_pts) <2 or len(target_pts) <2:
        raise ValueError("Source points and target points not enough.")
    if len(source_pts) != len(target_pts):
        raise ValueError("Source points and target points must have the same number of points.")
    if source_pts.shape[1] != 2 or target_pts.shape[1] != 2:
        raise ValueError("Source points and target points must be 2D coordinates.")
    
  # Prepare the output image
    height, width = image.shape[:2]
    output = np.zeros_like(image)
    # 计算控制点与目标点的最大曼哈顿距离作为阈值
     
    #manhattan_dists = np.abs(source_pts - target_pts).sum(axis=1)
    #threshold = 2*np.max(manhattan_dists)
    # 动态决定网格步长
    min_grid_size=10
    max_grid_size=20 
    """创建与图像尺寸匹配的网格，确保 source_pts 点在网格顶点上。""" 
    # 确定网格数量
    num_x = max(min(width // min_grid_size, width // max_grid_size), 1)
    num_y = max(min(height // min_grid_size, height // max_grid_size), 1)
    
    x_lines = np.linspace(0, width - 1, num_x, dtype=int)
    y_lines = np.linspace(0, height - 1, num_y, dtype=int)
    
        # 将source_pts添加到网格坐标中
    add_pts = np.array(source_pts, dtype=int)
    all_x = np.unique(np.concatenate((x_lines, add_pts[:, 0])))
    all_y = np.unique(np.concatenate((y_lines, add_pts[:, 1])))
    all_x_num=len(all_x)
    all_y_num=len(all_y)
    transformed_points = []

    start_time_a = time.time() 
    # 计算每个网格顶点的变形
    for y in all_y:
        for x in all_x:
            transformed_points.append(mls_affine(source_pts, target_pts, (x, y)))   
    transformed_points = np.array(transformed_points, dtype=int)

    end_time_a = time.time()
    elapsed_time_a = end_time_a - start_time_a
    print(f"顶点点变形消耗时间: {elapsed_time_a} 秒")


    start_time_b = time.time()
    # 网格内像素点变形
    for i in range(len(all_y) - 1):
        for j in range(len(all_x) - 1):
            # 原图像坐标
            src_quad = np.array([
                [all_x[j], all_y[i]],
                [all_x[j + 1], all_y[i]],
                [all_x[j], all_y[i + 1]],
                [all_x[j + 1], all_y[i + 1]]
            ], dtype=np.float32)

            # 变形后图像坐标
            dst_quad = np.array([
                transformed_points[i * len(all_x) + j],
                transformed_points[i * len(all_x) + j + 1],
                transformed_points[(i + 1) * len(all_x) + j],
                transformed_points[(i + 1) * len(all_x) + j + 1]
            ], dtype=np.float32)

            # 计算仿射变换矩阵
            M = cv2.getAffineTransform(src_quad[:3], dst_quad[:3])
            N = cv2.getAffineTransform(src_quad[1:], dst_quad[1:])
            # 生成 y 和 x 的网格坐标
            ys, xs = np.meshgrid(
                np.arange(all_y[i], all_y[i + 1]),
                np.arange(all_x[j], all_x[j + 1]),
                indexing='ij'
            )
            # 将坐标组合成形状为(N, 1, 2)的数组，以适应cv2.transform函数的要求
            coords = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)

            # 应用仿射变换
            transformed_coords = cv2.transform(coords, M).reshape(-1, 2)
            transformed_coords_opposite = cv2.transform(coords, N).reshape(-1, 2)

            # 提取变换后的x和y坐标
            pxs, pys = transformed_coords[:, 0], transformed_coords[:, 1]
            pxso, pyso = transformed_coords_opposite[:, 0], transformed_coords_opposite[:, 1]
            # 整数化并验证边界条件
            valid_indices = (
                (0 <= pys) & (pys < height) & 
                (0 <= pxs) & (pxs < width)
            )
            valid_indices_opposite = (
                (0 <= pyso) & (pyso < height) & 
                (0 <= pxso) & (pxso < width)
            )
            # 将像素从image复制到output
            output[pys[valid_indices].astype(int), pxs[valid_indices].astype(int)] = \
            image[ys.reshape(-1)[valid_indices], xs.reshape(-1)[valid_indices]]
            output[pyso[valid_indices_opposite].astype(int), pxso[valid_indices_opposite].astype(int)] = \
            image[ys.reshape(-1)[valid_indices_opposite], xs.reshape(-1)[valid_indices_opposite]]
            # 向量化避免for循环

    end_time_b = time.time()
    elapsed_time_b = end_time_b - start_time_b
    print(f"网格内像素点变形消耗时间: {elapsed_time_b} 秒")

    start_time_c = time.time()
    # for y in range(height):
    #     for x in range(width):
    #         if np.all(output[y, x] == 0):  # 如果像素为黑，则进行插值
    #             # 使用双线性插值
    #              output[y, x] = bilinear_interpolation(output, x, y)
    #中值滤波去除黑线
    output = remove_black_lines(output) 

    #output =  interpolate_black_lines(output)  # opencv去除黑线
    
    # 使用 inpaint 方法去除黑线
    ##mask = (output == 0).astype(np.uint8)[:, :, 0]  # 对黑色部分进行掩码提取
    #output = cv2.inpaint(output, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    #三次插值去除黑线
    output = cv2.resize(output, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_CUBIC)
    end_time_c = time.time()
    elapsed_time_c = end_time_c - start_time_c
    print(f"去黑线: {elapsed_time_c} 秒")

    #网格绘制
    #可视化网格：绘制开始
    start_time_d = time.time()
    output_grid = remove_black_lines(output)
    # 显示变形前网格
    for x in all_x:
        output_grid = cv2.line(output_grid, (x, 0), (x, height), (0, 255, 0), 1)
    for y in all_y:
        output_grid = cv2.line(output_grid, (0, y), (width, y), (0, 255, 0), 1)
    # 标记source_pts
    for pt in source_pts:
        output_grid = cv2.circle(output_grid, tuple(pt), 5, (0, 0, 255), -1)
    # 显示变形后网格
    for i in range(all_y_num):
        for j in range(all_x_num):
            # 检查 j 是否在有效范围内
            if j < all_x_num - 1:  # 确保 j 不会越界
                # 计算当前和下一个点的索引
                current_index = i * all_x_num + j
                next_index = i * all_x_num + (j + 1)
                if next_index < len(transformed_points):  # 确保 next_index 在范围内
                    cv2.line(output_grid, 
                            tuple(transformed_points[current_index]),
                            tuple(transformed_points[next_index]), 
                            (255, 0, 0), 1)

            # 检查 i 是否在有效范围内
            if i < all_y_num - 1:  # 确保 i 不会越界
                # 计算当前和下一个点的索引
                current_index = i * all_x_num + j
                next_row_index = (i + 1) * all_x_num + j
                if next_row_index < len(transformed_points):  # 确保 next_row_index 在范围内
                    cv2.line(output_grid, 
                            tuple(transformed_points[current_index]),
                            tuple(transformed_points[next_row_index]), 
                            (255, 0, 0), 1)
       
    # 显示变形后的source_pts
    for pt in source_pts:
        xx,yy = mls_affine(source_pts, target_pts, pt)
        output_grid = cv2.circle(output_grid, tuple([int(xx),int(yy)]), 5, (0, 255, 255), -1)  # 用红色标记
    output =     output_grid  # 显示网格
    end_time_d = time.time()
    elapsed_time_d = end_time_d - start_time_d 
    print(f"网格绘制: {elapsed_time_d} 秒")
    # 可视化网格：绘制结束

 
    return output  # 去除黑线


 
 
    #   # 使用并行处理
    # results = Parallel(n_jobs=-1)(delayed(process_pixel)(x, y) for x in range(width) for y in range(height))

    # for x_prime, y_prime, color in results:
    #     output[y_prime, x_prime] = color
    #return output
 

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
