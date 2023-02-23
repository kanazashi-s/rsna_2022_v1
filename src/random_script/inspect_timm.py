import os
from collections import OrderedDict
import timm
import torch
import torch.nn as nn
import torchinfo


if __name__ == "__main__":
    print(timm.list_models("*vit*", pretrained=True))
    model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=1, in_chans=3)
    print(torchinfo.summary(model, input_size=(1, 3, 224, 224), depth=4))

    model2 = timm.create_model("efficientnetv2_s", pretrained=False, in_chans=3)
    print(torchinfo.summary(model2, input_size=(1, 3, 224, 224), depth=4))
    model2.classifier = nn.Linear(model2.conv_head.out_channels, 1)

    print("a")