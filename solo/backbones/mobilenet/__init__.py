
from .mobilenet import mobilenetv2, mobilenetv3_large, mobilenetv3_small


def mobilenet_v2(method): #*args, **kwargs
    return mobilenetv2()

def mobilenet_v3_large(method): #*args, **kwargs
    return mobilenetv3_large()

def mobilenet_v3_small(method): #*args, **kwargs
    return mobilenetv3_small()

__all__ = ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]
