import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b3


class EfficientNetWrapper(nn.Module):
    """MNASNetWrapper Wrapper class for CIFAR10 classification"""
    def __init__(self, pretrained=False):
        super(EfficientNetWrapper, self).__init__()
        
        # Get Feature layers
        self.efficientnet = efficientnet_b3(pretrained=pretrained)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1536, 10)
        )
    
    def forward(self, x):
        return self.efficientnet(x)