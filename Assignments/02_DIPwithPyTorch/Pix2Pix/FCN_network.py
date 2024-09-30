import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # 输入通道：8，输出通道：16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 输入通道：16，输出通道：32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 输入通道：32，输出通道：64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
                
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.deconv3 = nn.Sequential(  # 反卷积层
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(  # 反卷积层
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(  # 反卷积层
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()#  
        )
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)       # 第一层卷积
        x2 = self.conv2(x1)      # 第二层卷积
        x3 = self.conv3(x2)      # 第三层卷积
        x4 = self.conv4(x3)      # 第四层卷积    # 第六层卷积，得到编码后的特征
        #print(f"x1 shape: {x1.shape}")
        #print(f"x2 shape: {x2.shape}")
        #print(f"x3 shape: {x3.shape}")
        #print(f"x4 shape: {x4.shape}")
        # Decoder forward pass 
        x = self.deconv3(x4)       # 第三层反卷积
        #print(f"第四层反卷积后 x shape: {x.shape}")
        x = self.deconv2(x)       # 第二层反卷积
        #print(f"第三层反卷积后 x shape: {x.shape}")
        x = self.deconv1(x)       # 第一层反卷积
        #print(f"第二层反卷积后 x shape: {x.shape}")
        output = self.final_conv(x)  # 输出层，得到最终的结果    
        #print(f"第一层反卷积后 x shape: {output.shape}")
        # Decoder forward pass

        ### FILL: encoder-decoder forward pass

        #output = x
        
        return output
    