import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
import os

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from train import GaussianTrainer, TrainConfig
from data_utils import ColmapDataset  # 导入 ColmapDataset 类

import sqlite3
import numpy as np

def get_image_data_by_name(dataset, image_name):
    """
    根据图像名称获取其索引，并调用 dataset 的 getitem 方法返回相应的数据
    """
    for idx, image_path in enumerate(dataset.image_paths):
        if os.path.basename(image_path) == image_name:
            return idx
    raise ValueError(f"Image {image_name} not found in dataset")


class ImagePairDataset(Dataset):
    def __init__(self, image_a_path, image_b_path):
        self.image_a = cv2.imread(image_a_path)
        self.image_b = cv2.imread(image_b_path)
        self.image_a = cv2.cvtColor(self.image_a, cv2.COLOR_BGR2RGB) / 255.0
        self.image_b = cv2.cvtColor(self.image_b, cv2.COLOR_BGR2RGB) / 255.0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'image_a': torch.tensor(self.image_a, dtype=torch.float32).permute(2, 0, 1),
            'image_b': torch.tensor(self.image_b, dtype=torch.float32).permute(2, 0, 1)
        }

def fine_tune_model(checkpoint_path, image_a_path, image_b_path, output_dir, num_epochs=100, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset
    dataset = ImagePairDataset(image_a_path, image_b_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = GaussianModel().to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            image_a = batch['image_a'].to(device)
            image_b = batch['image_b'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(image_a)

            # Compute losses
            l1_loss = nn.L1Loss()(output, image_b)
            ssim_loss = ssim(output, image_b)  # 假设已经定义了SSIM损失函数
            pos_loss = positional_loss(output, image_b)  # 位置损失
            arap_loss = arap_regularization(model)  # ARAP正则化
            mask_loss = adaptive_masking(output, image_b)  # 自适应掩码

            # Total loss
            total_loss = l1_loss + ssim_loss + pos_loss + arap_loss + mask_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")

    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'fine_tuned_model.pth'))

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_000012.pt"  # 修改为你的检查点路径
    image_a_path = "path/to/A.png"  # 修改为A.png的路径
    image_b_path = "path/to/B.png"  # 修改为B.png的路径
    output_dir = "fine_tuned_checkpoints"
    colmap_dir = "data/chair"
    
    # 获取图片名
    image_name = 'r_6.png'
    db_path = os.path.join(colmap_dir, 'database.db')
    
    # 获取相机参数
    # 示例用法  
    # 图像文件名是其在数据库中的序号
    dataset = ColmapDataset(colmap_dir)
    image_name = "r_99.png"
    image_data = get_image_data_by_name(dataset, image_name)
    print(image_data)
    
    #fine_tune_model(checkpoint_path, image_a_path, image_b_path, output_dir)