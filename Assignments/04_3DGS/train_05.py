import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
import os
import cv2
import torch.nn.functional as F
from pytorch_msssim import ssim
from pytorch3d.ops import knn_points, sample_farthest_points
from train import *
from sklearn.decomposition import PCA
from pytorch3d.transforms import quaternion_to_matrix,matrix_to_quaternion

# 定义位置损失函数
def positional_loss(rendered_images, target_images, lambda_=0.5, tile_size=16):
    # 将图像分割成瓦片，并计算每个瓦片的平均颜色
    def average_tiles(image, tile_size):
        n, c, h, w = image.shape
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        h += pad_h
        w += pad_w
        image = image.view(n, c, h // tile_size, tile_size, w // tile_size, tile_size)
        image = image.mean(dim=(3, 5))
        return image

    # 计算传输成本
    def compute_transport_cost(u, v, c_u, c_v, lambda_):
        device = u.device  # 确保所有张量在同一个设备上
        color_distance = torch.cdist(c_u, c_v, p=2).to(device) ** 2
        positional_distance = torch.cdist(u, v, p=2).to(device) ** 2
        return lambda_ * color_distance + (1 - lambda_) * positional_distance

    # 使用Sinkhorn算法计算密集的2D对应关系
    def sinkhorn_algorithm(cost_matrix, epsilon=0.1, num_iters=50):  # 减少迭代次数
        device = cost_matrix.device  # 确保所有张量在同一个设备上
        #print(f"Cost matrix shape: {cost_matrix.shape}")  # 添加调试代码
        n, m = cost_matrix.shape
        u = torch.zeros(n).to(device)
        v = torch.zeros(m).to(device)
        K = torch.exp(-cost_matrix / epsilon).to(device)
        for _ in range(num_iters):
            u = epsilon * (torch.logsumexp(K + v.unsqueeze(0), dim=1) - torch.logsumexp(K + u.unsqueeze(1), dim=0))
            v = epsilon * (torch.logsumexp(K + u.unsqueeze(1), dim=0) - torch.logsumexp(K + v.unsqueeze(0), dim=1))
        return K * torch.exp(u.unsqueeze(1) + v.unsqueeze(0))

    # 获取图像的2D位置和颜色
    device = rendered_images.device  # 确保所有张量在同一个设备上
    rendered_images = average_tiles(rendered_images, tile_size)
    target_images = average_tiles(target_images, tile_size)
    u = torch.stack(torch.meshgrid(torch.arange(rendered_images.shape[2]), torch.arange(rendered_images.shape[3])), dim=-1).float().view(-1, 2).to(device)
    v = torch.stack(torch.meshgrid(torch.arange(target_images.shape[2]), torch.arange(target_images.shape[3])), dim=-1).float().view(-1, 2).to(device)
    c_u = rendered_images.view(3, -1).permute(1, 0).to(device)
    c_v = target_images.view(3, -1).permute(1, 0).to(device)

    # 计算传输成本矩阵
    cost_matrix = compute_transport_cost(u, v, c_u, c_v, lambda_)
    transport_plan = sinkhorn_algorithm(cost_matrix)

    # 计算位置损失
    positional_loss_value = torch.sum(transport_plan * cost_matrix)
    return positional_loss_value

# 定义匹配损失函数
def matching_loss(rendered_images, target_images, lambda_l1=0.5, lambda_ssim=0.5):
    positional_loss_value = positional_loss(rendered_images, target_images, lambda_=0.5)
    l1_loss = F.l1_loss(rendered_images, target_images)
    ssim_loss = 1 - ssim(rendered_images, target_images, data_range=1.0)
    return positional_loss_value + lambda_l1 * l1_loss + lambda_ssim * ssim_loss


# 定义自适应刚性掩码
class AdaptiveRigidityMask(nn.Module):
    def __init__(self, num_nodes):
        super(AdaptiveRigidityMask, self).__init__()
        self.m_ij = nn.Parameter(torch.zeros(num_nodes, num_nodes))  # 初始化掩码参数
        self.m_ij_d = nn.Parameter(torch.zeros(num_nodes, num_nodes))  # 距离掩码
        self.m_ij_r = nn.Parameter(torch.zeros(num_nodes, num_nodes))  # 旋转掩码
        self.eta = 0.1  # 超参数，用于重置掩码

    def forward(self, kappa_ij, neighbors):
        # 计算掩码后的权重
        kappa_ij_m = self.masked_weight(kappa_ij, self.m_ij)
        kappa_ij_m_d = self.masked_weight(kappa_ij, self.m_ij_d)
        kappa_ij_m_r = self.masked_weight(kappa_ij, self.m_ij_r)

        # 周期性重置掩码
        self.reset_masks()

        return kappa_ij_m, kappa_ij_m_d, kappa_ij_m_r

    def masked_weight(self, kappa_ij, m_ij):
        return kappa_ij / kappa_ij.sum(dim=1, keepdim=True) * torch.sigmoid(m_ij)

    def reset_masks(self):
        self.m_ij = nn.Parameter(torch.sigmoid(self.m_ij).clamp(min=self.eta))
        self.m_ij_d = nn.Parameter(torch.sigmoid(self.m_ij_d).clamp(min=self.eta))
        self.m_ij_r = nn.Parameter(torch.sigmoid(self.m_ij_r).clamp(min=self.eta))




def total_loss(gaussian_params, initial_positions, anchor_positions, anchor_rotations, m_ij, m_ij_d, m_ij_r, K=10, gamma=1.0, coarse_stage=True):
    """
    计算总损失，包括ARAP正则化损失、掩码损失、旋转损失和距离损失。

    参数:
    gaussian_params: GaussianParameters 对象，包含高斯点的参数。
    initial_positions: torch.Tensor, 初始位置。
    anchor_positions: torch.Tensor, 锚点位置。
    anchor_rotations: torch.Tensor, 锚点旋转矩阵或四元数。
    m_ij: torch.Tensor, 掩码参数。
    m_ij_d: torch.Tensor, 距离掩码。
    m_ij_r: torch.Tensor, 旋转掩码。
    K: int, 近邻数量。
    gamma: float, RBF核的超参数。
    coarse_stage: bool, 是否为粗阶段。
    中间量：
    weights: torch.Tensor, 归一化权重。
    返回:
    arap_loss: torch.Tensor, ARAP正则化损失。
    mask_loss: torch.Tensor, 掩码损失。
    r_loss: torch.Tensor, 旋转损失。
    d_loss: torch.Tensor, 距离损失。
    """
    device = anchor_positions.device  # 确保所有张量在同一个设备上
    initial_positions = initial_positions.to(device)
    
    if coarse_stage:
        positions = anchor_positions.to(device)
        rotations = anchor_rotations.to(device)
    else:
        positions = gaussian_params.positions.to(device)
        rotations = gaussian_params.compute_covariance().to(device)
        
    print("Shape of rotations:", rotations.shape)

    # 如果旋转矩阵是四元数，转换为3x3矩阵
    if rotations.shape[-1] == 4:
        rotations = quaternion_to_matrix(rotations)
    
    # 计算K近邻
    knn = knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=K)
    knn_indices = knn[1].squeeze(0).to(device)  # (N, K)
   
    # 初始化损失
    arap_loss = 0.0
    
    # 计算ARAP损失
    mu_i = initial_positions.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
    mu_i_current = positions.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
    mu_i_neighbors = initial_positions[knn_indices]  # (N, K, 3)
    mu_i_neighbors_current = positions[knn_indices]  # (N, K, 3)
    R_i = rotations.unsqueeze(1).expand(-1, K, -1, -1)  # (N, K, 3, 3)
    
    # 计算未归一化权重
    distances = torch.norm(mu_i_current - mu_i_neighbors_current, dim=2)  # (N, K)
    weights_unnormalized = torch.exp(-gamma * distances ** 2)  # (N, K)
    
    # 归一化权重
    weights = weights_unnormalized / torch.sum(weights_unnormalized, dim=1, keepdim=True)  # (N, K)
    
    # 计算相对位置
    relative_position_initial = mu_i - mu_i_neighbors  # (N, K, 3)
    relative_position_current = torch.einsum('nkj,nkij->nki', relative_position_initial, R_i) - (mu_i_current - mu_i_neighbors_current)  # (N, K, 3)
    
    # 扩展 m_ij 以匹配 weights 的尺寸
    m_ij_expanded = m_ij.unsqueeze(1).expand(-1, K, -1)  # (N, K, num_nodes)
    
    # 累加损失
    arap_loss = torch.sum(weights.unsqueeze(-1) * m_ij_expanded * torch.norm(relative_position_current, dim=2, keepdim=True) ** 2)  # 标量
    
    # 归一化损失
    arap_loss /= positions.shape[0]
    
    # 计算掩码损失
    mask_loss = torch.sum(torch.abs(torch.sigmoid(m_ij)))

    # 计算旋转损失
    q_i = gaussian_params.rotations.unsqueeze(1)  # (N, 1, 4)
    q_j = anchor_rotations.unsqueeze(0)  # (1, M, 4)
    r_loss = torch.sum(weights.unsqueeze(-1) * m_ij_r.unsqueeze(1).expand(-1, K, -1) * torch.norm(q_i - q_j, dim=-1) ** 2)

    # 计算距离损失
    mu_i = gaussian_params.positions.unsqueeze(1)  # (N, 1, 3)
    mu_j = anchor_positions.unsqueeze(0)  # (1, M, 3)
    initial_distances = torch.norm(initial_positions.unsqueeze(1) - initial_positions, dim=-1) ** 2
    current_distances = torch.norm(mu_i - mu_j, dim=-1) ** 2
    d_loss = torch.sum(weights.unsqueeze(-1) * m_ij_d.unsqueeze(1).expand(-1, K, -1) * torch.abs(current_distances - initial_distances))

    return arap_loss, mask_loss, r_loss, d_loss

# 损失函数权重
lambda_arap = 1.0
lambda_rot = 1.0
lambda_dist = 1.0
lambda_mask = 1.0
lambda_pos = 1.0

class TrainingProcess:
    def __init__(self, config, device):
        self.device = device
        self.config = config

        # 数据集和数据加载器，用来读取视点信息
        data_path = "data/chair"
        self.dataset = ColmapDataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        # 模型、损失函数和优化器
        points3D_xyz = self.dataset.points3D_xyz.to(device)
        points3D_rgb = self.dataset.points3D_rgb.to(device)
        self.model = GaussianModel(points3D_xyz, points3D_rgb).to(device)

        # 初始化自适应刚性掩码
        self.adaptive_rigidity_mask = AdaptiveRigidityMask(self.model.positions.shape[0]).to(device)

        # 读取B.png
        image_b_path = "data/chair/edited_image/r_14_edited.png"    # 修改为B.png的路径
        image_b = cv2.imread(image_b_path)
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB) / 255.0
        image_b = cv2.resize(image_b, (100, 100))  # 压缩成100x100
        self.image_b = torch.tensor(image_b, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        # 获取B.png的尺寸
        self.image_b_height, self.image_b_width = self.image_b.shape[2], self.image_b.shape[3]

        # 模型、损失函数和优化器
        points3D_xyz = self.dataset.points3D_xyz.to(device)
        points3D_rgb = self.dataset.points3D_rgb.to(device)
        self.model = GaussianModel(points3D_xyz, points3D_rgb).to(device)
        self.renderer = GaussianRenderer(image_height=self.image_b_height, image_width=self.image_b_width).to(device)  # 使用B.png的尺寸

        # 读取原始模型
        checkpoint_path = "data/chair/checkpoints/checkpoint_000003.pt"  # 修改为你的检查点路径
        if checkpoint_path:
            # Initialize trainer
            self.trainer = GaussianTrainer(self.model, self.renderer, config, device)
     # Initialize optimizer
        optable_params = [
            {'params': [self.model.positions], 'lr': 0.0001, "name": "xyz"},  # 提高学习率
            {'params': [self.model.colors], 'lr': 0.05, "name": "color"},  # 提高学习率
            {'params': [self.model.opacities], 'lr': 0.1, "name": "opacity"},  # 提高学习率
            {'params': [self.model.scales], 'lr': 0.01, "name": "scaling"},  # 提高学习率
            {'params': [self.model.rotations], 'lr': 0.005, "name": "rotation"},  # 提高学习率
            {'params': [self.adaptive_rigidity_mask.m_ij, self.adaptive_rigidity_mask.m_ij_d, self.adaptive_rigidity_mask.m_ij_r], 'lr': 0.001, "name": "mask"}  # 添加掩码参数
        ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.001, eps=1e-15)
        
        # 使用学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # 读取原始图名字（id）
        image_name = 'r_14.png'  # 修改为你的图像名称
        image_idx = None
        for idx, image_path in enumerate(self.dataset.image_paths):
            if os.path.basename(image_path) == image_name:
                image_idx = idx
                break
        if (image_idx is None):
            raise ValueError(f"Image {image_name} not found in dataset")
        print(f"Image {image_name} found at index {image_idx}")
        # 获取 r_14.png 的相机参数
        image_data = self.dataset[image_idx]
        self.K = image_data['K'].unsqueeze(0).to(device)
        self.R = image_data['R'].unsqueeze(0).to(device)
        self.t = image_data['t'].unsqueeze(0).to(device).reshape(-1, 3)

        # 保存初始的高斯参数,用于arap loss计算
        self.initial_gaussian_params = self.model.get_gaussian_params()
        self.initial_positions = self.initial_gaussian_params.positions

    def generate_comparison_image(self, rendered_images, epoch=0):
        rendered = rendered_images.detach().cpu().numpy()
        gt = self.image_b.detach().cpu().numpy()
        comparison = np.concatenate([gt, rendered], axis=3)  # 拼接图像
        comparison = (comparison * 255).clip(0, 255).astype(np.uint8)  # 转换为 uint8 类型
        comparison = comparison[0].transpose(1, 2, 0)  # 调整维度顺序
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
         
        cv2.imwrite(f"05/SG_Image/comparison_{epoch+1:06d}.png", comparison)
         

    def generate_initial_comparison_image(self):
        gaussian_params = self.model.get_gaussian_params()
        rendered_images = self.trainer.renderer(
            gaussian_params.positions,
            gaussian_params.covariance,
            gaussian_params.colors,
            gaussian_params.opacities,
            self.K.squeeze(0),
            self.R.squeeze(0),
            self.t.squeeze(0)
        )
        rendered_images = rendered_images.permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, 3, 100, 100)
        rendered = rendered_images.detach().cpu().numpy()
        gt = self.image_b.detach().cpu().numpy()
        comparison = np.concatenate([gt, rendered], axis=3)  # 拼接图像
        comparison = (comparison * 255).clip(0, 255).astype(np.uint8)  # 转换为 uint8 类型
        comparison = comparison[0].transpose(1, 2, 0)  # 调整维度顺序
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
        cv2.imwrite("05/SG_Image/comparison_initial.png", comparison)

    def get_anchor_points(self, voxel_size=0.1, num_anchor_points=100):
        device = self.device  # 获取设备

        # 获取高斯点的位置
        gaussian_positions = self.model.positions.detach().to(device)

        # 体素化3D场景并计算每个体素中3D高斯的质心
        min_coords = gaussian_positions.min(dim=0)[0]
        max_coords = gaussian_positions.max(dim=0)[0]
        voxel_grid = torch.floor((gaussian_positions - min_coords) / voxel_size).int()
        unique_voxels, inverse_indices = torch.unique(voxel_grid, return_inverse=True, dim=0)
        voxel_centers = torch.zeros((unique_voxels.shape[0], 3), device=device)
        for i in range(unique_voxels.shape[0]):
            voxel_centers[i] = gaussian_positions[inverse_indices == i].mean(dim=0)

        # 对稠密点云应用最远点采样（FPS）来下采样得到初始anchor点
        dense_point_cloud = voxel_centers.unsqueeze(0)  # (1, N, 3)
        sampled_points, _ = sample_farthest_points(dense_point_cloud, K=num_anchor_points)
        anchor_positions = sampled_points.squeeze(0).to(device)  # (Na, 3)

        # 计算每个锚点的旋转矩阵
        anchor_rotations = torch.zeros((num_anchor_points, 3, 3), device=device)
        for i in range(num_anchor_points):
            # 获取锚点的近邻点
            if gaussian_positions.shape[0] < 4:
                continue
            knn = knn_points(anchor_positions[i].unsqueeze(0).unsqueeze(0), gaussian_positions.unsqueeze(0), K=4)
            knn_indices = knn.idx.squeeze(0).squeeze(0)  # (4,)
            knn_positions = gaussian_positions[knn_indices]  # (4, 3)

            # 检查是否有足够的邻居
            if knn_positions.shape[0] < 4:
                continue

            # 计算协方差矩阵
            centered_knn_positions = knn_positions - knn_positions.mean(dim=0)
            covariance_matrix = centered_knn_positions.t() @ centered_knn_positions

            # 计算旋转矩阵（使用SVD分解）
            u, _, v = torch.svd(covariance_matrix)
            rotation_matrix = u @ v.t()
            anchor_rotations[i] = rotation_matrix

        return anchor_positions, anchor_rotations

    def linear_blend_skinning(self, gaussian_positions):
            """
            使用线性混合蒙皮（LBS）方法推导高斯点的变形场。
            
            参数:
            gaussian_positions: torch.Tensor, 高斯点的位置。
            
            返回:
            deformed_positions: torch.Tensor, 变形后的高斯点位置。
            deformed_quaternions: torch.Tensor, 变形后的高斯点旋转四元数。
            """
            device = self.device  # 获取设备
            gaussian_positions = gaussian_positions.to(device)
            self.anchor_positions = self.anchor_positions.to(device)
            self.anchor_rotations = self.anchor_rotations.to(device)

            # 计算K近邻
            knn = knn_points(gaussian_positions.unsqueeze(0), self.anchor_positions.unsqueeze(0), K=4)
            knn_indices = knn.idx.squeeze(0)  # (N, 4)
            knn_distances = knn.dists.squeeze(0)  # (N, 4)

            # 计算权重
            weights = torch.exp(-knn_distances)
            weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化权重

            # 获取K近邻的锚点位置和旋转矩阵
            anchor_positions_knn = self.anchor_positions[knn_indices]  # (N, 4, 3)
            anchor_rotations_knn = self.anchor_rotations[knn_indices]  # (N, 4, 3, 3)

            # 计算变形后的高斯点位置
            gaussian_positions_expanded = gaussian_positions.unsqueeze(1).expand_as(anchor_positions_knn)  # (N, 4, 3)
            deformed_positions = torch.sum(weights.unsqueeze(-1) * (anchor_rotations_knn @ (gaussian_positions_expanded - anchor_positions_knn).unsqueeze(-1)).squeeze(-1) + anchor_positions_knn, dim=1)

            # 计算变形后的旋转矩阵
            weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (N, 4, 1, 1)
            deformed_rotations = torch.sum(weights_expanded * anchor_rotations_knn, dim=1)  # (N, 3, 3)
            
            # 保证旋转矩阵的正交性
            u, _, v = torch.svd(deformed_rotations)
            deformed_rotations = torch.einsum('nij,njk->nik', u, v.transpose(-2, -1))

            # 将旋转矩阵转换为四元数
            deformed_quaternions = matrix_to_quaternion(deformed_rotations)

            return deformed_positions, deformed_quaternions

    def train(self):
        num_epochs = 100
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # 获取当前高斯参数
            gaussian_params = self.model.get_gaussian_params()
            
            # 使用LBS方法计算变形后的高斯点位置和旋转矩阵
            
            # 渲染图像
            rendered_images = self.trainer.renderer(
                gaussian_params.positions,
                gaussian_params.covariance,
                gaussian_params.colors,
                gaussian_params.opacities,
                self.K.squeeze(0),
                self.R.squeeze(0),
                self.t.squeeze(0)
            )

            #self.anchor_positions, self.anchor_rotations = self.get_anchor_points()
            self.anchor_positions, self.anchor_rotations = self.model.positions, self.model.rotations
            # 计算损失
            rendered_images = rendered_images.permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, 3, 100, 100) 对齐被编辑图片
            arap_loss, mask_loss, r_loss, d_loss = total_loss(
                self.model.get_gaussian_params(),
                self.initial_positions,
                self.anchor_positions,
                self.anchor_rotations,
                self.adaptive_rigidity_mask.m_ij,
                self.adaptive_rigidity_mask.m_ij_d,
                self.adaptive_rigidity_mask.m_ij_r
            )
            loss = matching_loss(rendered_images, self.image_b, lambda_l1=0.5, lambda_ssim=0.5) + \
                   arap_loss + r_loss + d_loss + mask_loss  # 添加掩码损失、旋转损失和距离损失

            # 反向传播和优化
            self.trainer.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.trainer.model.parameters(), 
                self.trainer.config.grad_clip
            )
                
            self.trainer.optimizer.step()
            self.scheduler.step()  # 更新学习率
            
            # 重置掩码
            with torch.no_grad():
                self.adaptive_rigidity_mask.m_ij.data = torch.max(self.adaptive_rigidity_mask.m_ij.data, torch.tensor(0.1).to(self.device))
                self.adaptive_rigidity_mask.m_ij_d.data = torch.max(self.adaptive_rigidity_mask.m_ij_d.data, torch.tensor(0.1).to(self.device))
                self.adaptive_rigidity_mask.m_ij_r.data = torch.max(self.adaptive_rigidity_mask.m_ij_r.data, torch.tensor(0.1).to(self.device))
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

            # 每5个周期保存一次检查点和对比图片
            if (epoch + 1) % 5 == 0:
                # 保存检查点
                checkpoint_path = os.path.join("05/SG_checkpoints", f"checkpoint_{epoch+1:06d}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)

                # 保存对比图片
                self.generate_comparison_image(rendered_images, epoch)

        print("Training completed!")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TrainConfig()
    training_process = TrainingProcess(config, device)
    training_process.generate_initial_comparison_image()
    training_process.train()

if __name__ == "__main__":
    main()