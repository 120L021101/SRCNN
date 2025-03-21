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

class ResBlock(nn.Module):
    def __init__(self, num_channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class ResSR(nn.Module):
    def __init__(self, num_channels=1, num_res=16):
        super(ResSR, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.relu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(num_res)])
        self.conv2 = nn.Conv2d(64, num_channels, kernel_size=9, padding=9 // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        residual = self.res_blocks(x)
        x = self.conv2(residual + x)
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

    def forward(self, sr, hr):
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)

        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        if self.resize:
            sr = F.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            hr = F.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)

        sr_vgg = self.vgg_layers(sr)
        hr_vgg = self.vgg_layers(hr)

        return F.mse_loss(sr_vgg, hr_vgg)