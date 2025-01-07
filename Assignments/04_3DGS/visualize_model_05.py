import torch
from pathlib import Path
import cv2
import numpy as np
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from data_utils import ColmapDataset
from train import GaussianTrainer, TrainConfig
import random

def generate_images_from_checkpoint(checkpoint_path: str, output_dir: str, colmap_dir: str, device: str = 'cuda'):
    # Set device生成 预训练模型和训练后模型渲染图比较
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset
    dataset = ColmapDataset(colmap_dir)
    sample = dataset[0]['image']
    H, W = sample.shape[:2]

    # Initialize model using COLMAP points
    model = GaussianModel(
        points3D_xyz=dataset.points3D_xyz,
        points3D_rgb=dataset.points3D_rgb
    )

    # Initialize renderer
    renderer = GaussianRenderer(
        image_height=H,
        image_width=W
    )

    # Create a dummy config
    config = TrainConfig()

    # Initialize trainer
    trainer = GaussianTrainer(model, renderer, config, device)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Generate images
    max_images = 5  # 只生成五张图片
    for i in range(max_images):
        data_item = random.choice(dataset)
        images = data_item['image'].unsqueeze(0).to(device)
        K = data_item['K'].unsqueeze(0).to(device)
        R = data_item['R'].unsqueeze(0).to(device)
        t = data_item['t'].unsqueeze(0).to(device)

        batch = {
            'image': images,
            'K': K,
            'R': R,
            't': t
        }

        with torch.no_grad():
            rendered_images = trainer.train_step(batch, in_train=False)

        trainer.save_debug_images(
            epoch=0,
            rendered_images=rendered_images,
            gt_images=images,
            image_paths=[data_item['image_path']]
        )

if __name__ == "__main__":
    checkpoint_path = "05/SG_checkpoints/checkpoint_000100.pt"
    output_dir = "output/05images"
    colmap_dir = "data/chair"
    generate_images_from_checkpoint(checkpoint_path, output_dir, colmap_dir)