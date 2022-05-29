import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152


class ResNet152Wrapper(nn.Module):
    """ResNet-152 Wrapper class for CIFAR10 classification"""
    def __init__(self, pretrained=False):
        super(ResNet152Wrapper, self).__init__()
        
        # Get Feature layers
        self.resnet152 = resnet152(pretrained=pretrained)
        self.resnet152.fc = nn.Linear(in_features=2048, out_features=10)
    
    def forward(self, x):
        return self.resnet152(x)
    
    def get_named_modules(self):
        return [item for item in self.named_modules()]
    
    def get_intermediate_output(self, extracted_layers, x):
        outputs = []
        for name, module in self.resnet152._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in extracted_layers:
                outputs.append(x)
        return torch.cat(outputs, dim=1)if len(outputs) > 1 else outputs[0]