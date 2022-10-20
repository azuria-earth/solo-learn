
from .efficientnet import efficientnet_lite0 as lite0_efficientnet 

def efficientnet_lite0(method): #*args, **kwargs
    return lite0_efficientnet()

__all__ = ["efficientnet_lite0"]
