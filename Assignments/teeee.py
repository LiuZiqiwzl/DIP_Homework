import torch
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))

# 检查 cuDNN 版本
print("cuDNN version:", torch.backends.cudnn.version())

# 检查 cuDNN 是否可用
from torch.backends import cudnn
print("cuDNN is available:", cudnn.is_available())

# 检查 cuDNN 是否接受 CUDA tensor
a = torch.tensor(1.0)
print("cuDNN is acceptable for CUDA tensor:", cudnn.is_acceptable(a.cuda()))
