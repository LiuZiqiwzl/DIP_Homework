import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss

from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import *
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


# ================ 训练单个 epoch 函数 =================
def train_one_epoch(generator, discriminator, train_loader, criterion_gan,  criterion_l1,optimizer_g, optimizer_d, device, epoch, num_epochs, lambda_l1=2):
    """
    Train one epoch for Pix2Pix (cGAN).

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        train_loader (DataLoader): Training dataset loader.
        criterion_gan (nn.BCELoss or similar): GAN loss function.
        criterion_l1 (nn.L1Loss): L1 loss function for reconstruction.
        optimizer_g (Optimizer): Optimizer for generator (e.g., Adam).
        optimizer_d (Optimizer): Optimizer for discriminator (e.g., Adam).
        device (torch.device): Training device (GPU/CPU).
        epoch (int): The current epoch.
        num_epochs (int): Total number of epochs.
        lambda_l1 (float): Weight for L1 loss in generator loss (default 100, as per Pix2Pix paper).

    Returns:
        tuple: Average discriminator loss and generator loss for the epoch.
    """
    generator.train()
    discriminator.train()

    running_d_loss = 0.0
    running_g_loss = 0.0

    for i, ( image_rgb,image_semantic) in enumerate(train_loader):
                # Move data to device (GPU/CPU)
        image_semantic = image_semantic.to(device)  # Conditional input
        image_rgb = image_rgb.to(device)  # Target RGB image
        batch_size = image_semantic.size(0)

        # -------------------------------
        # Train Discriminator
        # -------------------------------
        optimizer_d.zero_grad()

        # Generate fake images
        fake_image_rgb = generator(image_semantic)

        # Discriminator outputs for real and fake pairs
        real_pair = torch.cat((image_semantic, image_rgb), dim=1)  # [B, C_cond + C_img, H, W]
        fake_pair = torch.cat((image_semantic, fake_image_rgb.detach()), dim=1)

        real_output = discriminator(real_pair)  # 判别 (条件图像, 真实图像)
        fake_output = discriminator(fake_pair)  # 判别 (条件图像, 生成图像)

        # Create real and fake labels
        real_labels = torch.ones_like(real_output, device=device)  # 判别真实样本目标为 1
        fake_labels = torch.zeros_like(fake_output, device=device)  # 判别伪造样本目标为 0

        # Compute discriminator loss
        d_loss_real = criterion_gan(real_output, real_labels)
        d_loss_fake = criterion_gan(fake_output, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5  # Average real and fake loss
        d_loss.backward()
        optimizer_d.step()

        running_d_loss += d_loss.item()

        # -------------------------------
        # Train Generator
        # -------------------------------
        optimizer_g.zero_grad()

        # Note: We do not detach fake_image_rgb because generator needs gradients
        fake_pair_for_g = torch.cat([image_semantic, fake_image_rgb], dim=1)  # 拼接条件 + 生成的图像
        fake_output_for_g = discriminator(fake_pair_for_g)

        # Generator GAN loss: wants discriminator to output 1 for fake samples
        g_gan_loss = criterion_gan(fake_output_for_g, real_labels)

        # Generator L1 loss: penalize reconstruction error between generated and target images
        g_l1_loss = criterion_l1(fake_image_rgb, image_rgb)

        # Compute total generator loss
        g_loss = g_gan_loss + lambda_l1 * g_l1_loss
        g_loss.backward()
        optimizer_g.step()

        running_g_loss += g_loss.item()

        # -------------------------------
        # Logging
        # -------------------------------
        if i % 20 == 19:
            avg_d_loss = running_d_loss / (i + 1)
            avg_g_loss = running_g_loss / (i + 1)
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}'
            )
        save_images(image_semantic, image_rgb, fake_image_rgb, 'Gan_train_results/', epoch)


    # Compute average losses for the epoch
    epoch_d_loss = running_d_loss / len(train_loader)
    epoch_g_loss = running_g_loss / len(train_loader)



    return epoch_d_loss, epoch_g_loss
    

# 验证函数
def validate(generator, val_loader, device, epoch, folder_name='Gan_validation_results', num_images=5):
    generator.eval()
    running_loss = 0.0

    os.makedirs(folder_name, exist_ok=True)
    with torch.no_grad():
        for i, (real_images,image_semantic) in enumerate(val_loader):
            real_images = real_images.to(device)
            image_semantic = image_semantic.to(device)
            fake_images = generator(image_semantic)

            # 保存生成图像
            if i % 5 == 0:
                save_images(image_semantic, real_images, fake_images, folder_name, epoch, num_images)

    print(f'Validation completed for Epoch {epoch + 1}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


    # 实例化模型
    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)


    # 检查是否存在模型检查点文件
    generator_path = os.path.join('Gan_checkpoints', 'generator_epoch_begin.pth')
    discriminator_path = os.path.join('Gan_checkpoints', 'discriminator_epoch_begin.pth')
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path))
        print("Generator model loaded from generator_epoch_begin.pth")
    else:
        print("No pre-trained generator found, initializing a new one.")

    if os.path.exists(discriminator_path):
        discriminator.load_state_dict(torch.load(discriminator_path))
        print("Discriminator model loaded from discriminator_epoch_begin.pth")
    else:
        print("No pre-trained discriminator found, initializing a new one.")

    criterion_gan = nn.BCELoss()
    criterion_l1 = L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    scheduler_g = StepLR(optimizer_g, step_size=64, gamma=0.4)
    scheduler_d = StepLR(optimizer_d, step_size=64, gamma=0.6)

    num_epochs = 100
    # 打印模型结构（可选）
    #print(generator)
    #print(discriminator)


    for epoch in range(num_epochs):
        train_d_loss, train_g_loss = train_one_epoch(
            generator, discriminator, train_loader, criterion_gan, criterion_l1,optimizer_g, optimizer_d, device, epoch, num_epochs
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed! "
            f"Avg D Loss: {train_d_loss:.4f}, Avg G Loss: {train_g_loss:.4f}")
    
        # 验证模型  
        if epoch % 5 == 0:
            validate(generator, val_loader, device, epoch)

        scheduler_g.step()
        scheduler_d.step()

        # 保存模型检查点
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f'Gan_checkpoints/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'Gan_checkpoints/discriminator_epoch_{epoch+1}.pth')
if __name__ == '__main__':
    main()
