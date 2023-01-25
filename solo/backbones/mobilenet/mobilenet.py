import torch
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

def mobilenet_v2():
    return mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2', progress=True)

def mobilenet_v3_large():
    return mobilenet_v3_large(weights='IMAGENET1K_V1', progress=True)

def mobilenet_v3_small():
    return mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1', progress=True)

__all__ = ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]
