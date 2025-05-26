import torch.nn as nn

class ClassificationHead(nn.Module):
    """基于分割网络最后一层特征做图像级分类"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        # x: [B, C, H, W]  → [B, C, 1, 1] → [B, C] → [B, num_classes]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
