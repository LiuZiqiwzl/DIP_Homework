import cv2
import numpy as np
import gradio as gr
from scipy.interpolate import RBFInterpolator
import numpy as np
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import pairwise_distances  


def compute_weights(p, points,  alpha=1.0, eps=1e-8):
        dists_squared = np.sum((points - p )** 2, axis=1)
        weights = (1 / (dists_squared + eps))**alpha
        return weights

def mls_affine(source_pts, target_pts, p,  alpha=1.0, eps=1e-8):
    """
    Moving least squares (MLS) deformation with affine transformations.
    """
    weights = compute_weights(p, source_pts, alpha, eps)
    # 计算控制点对该点的影响力计算 并归一化
    Norm_Weights=weights / np.sum(weights)

    # p*，q*
    centroid_source = np.sum(source_pts * Norm_Weights[:, np.newaxis], axis=0)
    centroid_target = np.sum(target_pts * Norm_Weights[:, np.newaxis], axis=0)

    # Move points to the centroid, p^i,q^i
    source_demean = source_pts - centroid_source
    target_demean = target_pts - centroid_target

    # Compute weighted covariance matrix
    P = (source_demean * weights[:, np.newaxis]).T
    #Q = target_demean * weights[:, np.newaxis]
    T = P@ source_demean #左矩阵
    V = P @ target_demean #右矩阵
        
    #    pinv广义逆
    M = np.linalg.pinv(T) @ V
    # v'= fa(v)=q*+(v-p*)@M
    p_prime = centroid_target +   (p - centroid_source) @ M
    return p_prime
  
def adaptive_median_filter(image, max_kernel_size=5):
    """
    对图像应用自适应中值滤波。
    
    参数:
        image (np.array): 原始图像的NumPy数组。
        max_kernel_size (int): 滤波器的最大窗口大小。
        
    返回:
        result (np.array): 滤波处理后的图像。
    """ 

    def get_median_filtered_region(x, y, window_size):
        half_size = window_size // 2
        region = padded_image[x-half_size:x+half_size+1, y-half_size:y+half_size+1]
        return np.median(region)
    
    padded_image = np.pad(image, ((max_kernel_size//2,)*2, (max_kernel_size//2,)*2, (0,0)), 'symmetric') 
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window_size = 3
            while window_size <= max_kernel_size:
                region = padded_image[i:i+window_size, j:j+window_size]
                min_val, med_val, max_val = np.min(region), np.median(region), np.max(region)
                
                if min_val < med_val < max_val:
                    image[i, j] = med_val
                    break
                else:
                    window_size += 2
                    
            if window_size > max_kernel_size:
                image[i, j] = med_val
    
    return image

def remove_black_lines(image):
    """
    该函数接受一个图像的NumPy数组作为输入，返回一个处理后的图像，努力去除变形产生的黑线。

    参数:
        image (np.array): 原始图像的NumPy数组。

    返回:
        result (np.array): 黑线被去除后的图像。
    """

    if len(image.shape) == 2:
            # 图像是灰度的
        is_grayscale = True
    elif len(image.shape) == 3:
        # 图像是彩色的
        is_grayscale = False
    else:
        raise ValueError("Unsupported image format")
  
    # 创建掩码，标记出黑色区域（值为0的区域）
    if is_grayscale:
        mask = (image == 0)
    else:
        mask = np.all(image == 0, axis=-1)

    # 中值滤波处理
    if is_grayscale:
        # 单通道中值滤波
        median_filtered = cv2.medianBlur(image, 3)
        image[mask] = median_filtered[mask]
    else:
        # 对每个通道分别进行中值滤波
        for channel in range(image.shape[2]):
            median_filtered_channel = cv2.medianBlur(image[:, :, channel], 3)
            channel_mask = mask[:, :, channel] if mask.ndim == 3 else mask
            image[channel_mask, channel] = median_filtered_channel[channel_mask]

    return image
 
def interpolate_black_lines(image):
    """
    使用插值方法解决图像中的黑线问题。
    """
    if len(image.shape) == 2:
        # 灰度图像
        is_grayscale = True
    elif len(image.shape) == 3:
        # 彩色图像
        is_grayscale = False
    else:
        raise ValueError("Unsupported image format")

    h, w = image.shape[:2] 

    # 创建掩码，标记黑线位置
    if is_grayscale:
        mask = (image == 0)
    else:
        mask = np.all(image == 0, axis=-1)

    # 使用局部平均来修复黑线
    kernel_size = 2  # 或其他适合图像的大小
    padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REFLECT)

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                # 提取邻域
                patch = padded_image[y:y + kernel_size, x:x + kernel_size]
                # 使用非空的邻域点做平均
                valid_values = patch[patch != 0]
                if valid_values.size > 0:
                    image[y, x] = np.mean(valid_values)

    return image