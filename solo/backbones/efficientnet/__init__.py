
from .efficientnet import efficientnet_lite0 as default_efficientnet


def efficientnet_lite0(method): #*args, **kwargs
    return default_efficientnet()