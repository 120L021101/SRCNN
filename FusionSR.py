
from torch import nn
from timm.models.vision_transformer import VisionTransformer
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
'''
CNN提取浅层特征
'''
class CNN_Features(nn.Module):
    def __init__(self):
        super(CNN_Features, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.unfold(x, kernel_size=8, stride=8)
        x = x.transpose(1, 2)
        return x


'''
Transformer提取中远距离特征
'''
class Transformer_Features(nn.Module):
    def __init__(self, channels):
        super(Transformer_Features, self).__init__()
        self.transformer = VisionTransformer(img_size=64, patch_size=8, num_classes=0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.transformer.forward_features(x)
        x = x[:, 1:, :] # 去掉句子头的特征，用来分类的那个, start of the sentence?
        return x

'''
拼接两个特征
'''
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.cnn_features = CNN_Features()
        self.transformer_features = Transformer_Features(channels=32)

    def forward(self, x):
        cnn_feats = self.cnn_features(x)
        transformer_feats = self.transformer_features(x)
        print(cnn_feats.shape, transformer_feats.shape)
        combined_feats = torch.cat((cnn_feats, transformer_feats), dim=2)
        return combined_feats


'''
从特征中重构图片
'''
class FinalReconstruction(nn.Module):
    def __init__(self):
        super(FinalReconstruction, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self.trans = nn.Linear(2816, 2048)

    def forward(self, x):
        x = self.trans(x)
        B, P, D = x.shape
        C = 3
        x = x.view(B, 32, 64, 64)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


'''
最终模型结构: FusionSR
'''
class FusionSR(nn.Module):
    def __init__(self):
        super(FusionSR, self).__init__()
        self.feature_fusion = FeatureFusion()
        self.reconstruction = FinalReconstruction()

    def forward(self, x):
        combined_feats = self.feature_fusion(x)
        print(combined_feats.shape)
        output = self.reconstruction(combined_feats)
        return output




def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor() 
    ])
    
    img = Image.open(image_path).convert("RGB") 
    img_tensor = transform(img).unsqueeze(0) 
    return img_tensor, img.size 


def save_image(tensor, output_path):
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = np.clip(tensor, 0, 1)  
    tensor = (tensor * 255).astype(np.uint8) 

    img = Image.fromarray(np.transpose(tensor, (1, 2, 0)))
    img.save(output_path, format="BMP") 


# sample test
if __name__ == "__main__":
    input_data, input_size = load_image("./data/butterfly_GT.bmp")
    print(input_data.shape)
    # input_data = torch.randn(2, 3, 64, 64)
    model = FusionSR()
    output = model(input_data)

    print(output.shape)

    save_image(output, "data/hello.bmp")