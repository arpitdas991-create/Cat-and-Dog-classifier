from torchvision.models import resnet18,ResNet18_Weights
import torch.nn as nn

def get_model(num_classes):
    model = resnet18(weights= None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model