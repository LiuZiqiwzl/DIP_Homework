import torch
import pytorch3d
 
def test_pytorch3d():
    print("PyTorch3D imported successfully!")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Testing PyTorch3D with GPU...")
        
        # 创建一个 tensor 并移动到 GPU
        x = torch.tensor([[1.0, 2.0, 3.0]], device=device)  # 确保点是二维的
        
        from pytorch3d.transforms import RotateAxisAngle
        rot = RotateAxisAngle(angle=90, axis="X").to(device)  # 将变换移动到 GPU
        x_transformed = rot.transform_points(x)
        
        print("Tensor on GPU: ", x)
        print("Transformed Tensor: ", x_transformed)
    else:
        print("CUDA is not available. Please check your PyTorch and GPU settings.")
 
test_pytorch3d()