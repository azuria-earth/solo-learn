

from efficientnet_lite_pytorch import EfficientNet
from timm.models.registry import register_model

from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

@register_model
def efficientnet_lite0():
    weights_path = EfficientnetLite0ModelFile.get_model_file_path()
    lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path)

    return lite0_model

__all__ = ["efficientnet_lite0"]


