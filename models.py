import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.optical_flow.raft import ResidualBlock


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


class Discriminator(nn.Module):
    def __init__(self, num_channels=1, input_size=33):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        with torch.no_grad():
            dummy_in = torch.zeros(1, num_channels, input_size, input_size)
            dummy_out = self.conv_layers(dummy_in)
            self.flattened_dim = dummy_out.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            # Comment the following line if you wish to use WGAN
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class VGGLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg19.children())[:36])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, sr, lr):
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = lr.repeat(1, 3, 1, 1)

        sr = (sr - self.mean) / self.std
        lr = (lr - self.mean) / self.std

        if self.resize:
            sr = F.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            lr = F.interpolate(lr, size=(224, 224), mode='bilinear', align_corners=False)

        sr_vgg = self.vgg_layers(sr)
        lr_vgg = self.vgg_layers(lr)

        return F.l1_loss(sr_vgg, lr_vgg)


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
        x_t = x
        for block in self.blocks:
            x_t = block(x_t)
        x = x + x_t

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