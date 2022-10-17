

from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite_pytorch.utils import get_same_padding_conv2d
import torch
import math
from torch.nn import functional as F
from timm.models.registry import register_model


from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

class Conv2dSamePadding(torch.nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



@register_model
def efficientnet_lite0():
    weights_path = EfficientnetLite0ModelFile.get_model_file_path()
    in_channels = 12
    lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', in_channels = in_channels, weights_path = weights_path)

    for child in lite0_model.children():

          for param in child.parameters():
            param.requires_grad = False

    in_channels = lite0_model._conv_head.in_channels
    out_channels = lite0_model._conv_head.out_channels
    num_ftrs = lite0_model._fc.in_features
    num_classes = 1000

    lite0_model._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)#= get_same_padding_conv2d(image_size=image_size)
    lite0_model._bn1 = torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.010000000000000009, eps = 0.001)
    lite0_model._fc = torch.nn.Linear(num_ftrs, num_classes)

    return lite0_model

@register_model
def efficientnet():
    from efficientnet_pytorch import EfficientNet
    #model = EfficientNet.from_pretrained('efficientnet-b0')
    model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)

    # for child in model.children():

    #       for param in child.parameters():
    #         param.requires_grad = False

    # in_channels = model._conv_head.in_channels
    # out_channels = model._conv_head.out_channels
    # num_ftrs = model._fc.in_features
    # num_classes = 1000

    # model._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)#= get_same_padding_conv2d(image_size=image_size)
    # model._bn1 = torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.010000000000000009, eps = 0.001)
    # model._fc = torch.nn.Linear(num_ftrs, num_classes)

    return model


__all__ = ["efficientnet_lite0", "efficientnet"]


