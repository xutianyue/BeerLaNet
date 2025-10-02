import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, densenet121
from src.colornorm import ColorNorm
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, d_channels):
        super().__init__()
        self.d_channels = d_channels
    def forward(self, x):
        return x.view(x.size(0), self.d_channels, 1, 1)

class CustomModel(nn.Module):
    def __init__(self, backbone, d_channels, num_classes):
        super(CustomModel, self).__init__()
        self.color_norm_layer = ColorNorm(r=d_channels, c=3, learn_S_init=True)
        self.adapt_conv = nn.Conv2d(d_channels, 3, kernel_size=1)  # add a 1x1 convolution to adapt input channels for backbones

        if backbone == 'resnet18':
            self.base_model = resnet18(pretrained=True)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif backbone == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif backbone == 'densenet121':
            self.base_model = densenet121(pretrained=True)
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    def forward(self, x):
        _, _, D = self.color_norm_layer(x, S=None, D=None, n_iter=10, unit_norm_S=True)
        d = self.adapt_conv(D)
        x = self.base_model(d)
        return x, d

def load_model(backbone, model_head, num_classes, device):
    if backbone not in ['resnet18', 'resnet50', 'densenet121']:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if model_head not in ['colornorm', 'none']:
        raise ValueError(f"Unsupported model head: {model_head}")
    model = None
    if model_head == 'colornorm':
        model = CustomModel(backbone=backbone, d_channels=8, num_classes=num_classes)
    elif model_head == 'none':
        if backbone == 'resnet18':
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == 'resnet50':
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == 'densenet121':
            model = densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    if model is None:
        raise ValueError(f"Failed to create model with backbone={backbone}, head={model_head}")
    return model.to(device)


