import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes: int):
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m