from torch import nn

import torch
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

class TransformerSRCNN(nn.Module):
    def __init__(self, num_channels=1, dim=64, num_heads=8, num_blocks=6, ff_dim=256):
        super(TransformerSRCNN, self).__init__()
        
        # Patch embedding: 将输入图像转换为序列
        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=3, stride=1, padding=1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim) for _ in range(num_blocks)
        ])
        
        # Final convolution to map back to the original channels
        self.final_conv = nn.Conv2d(dim, num_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # Step 1: Patch embedding
        # 将输入的 [batch_size, channels, height, width] 转换为 [batch_size, dim, height, width]
        x = self.patch_embed(x)
        
        # 将 [batch_size, dim, height, width] 展平为 [batch_size, height*width, dim] 作为 Transformer 的输入
        batch_size, dim, height, width = x.size()
        x = x.flatten(2).transpose(1, 2)  # 将形状变为 [batch_size, height*width, dim]
        
        # Step 2: Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Step 3: Reverse flattening to reconstruct the feature map
        x = x.transpose(1, 2).reshape(batch_size, dim, height, width)
        
        # Step 4: 最后的卷积层
        x = self.final_conv(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        
        # Feed-forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)  # Residual connection + norm
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual connection + norm
        
        return x

