
from .efficientnet import efficientnet as default_efficientnet
from .efficientnet import efficientnet_lite0 as lite0_efficientnet 


def efficientnet_lite0(method): #*args, **kwargs
    return lite0_efficientnet()

def efficientnet(method): #*args, **kwargs
    return default_efficientnet()

__all__ = ["efficientnet", "efficientnet_lite0"]
