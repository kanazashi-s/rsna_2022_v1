import os
from collections import OrderedDict
import timm
import torch
import torch.nn as nn
import torchinfo


if __name__ == "__main__":
    print(timm.list_models("*vit*", pretrained=True))
    model = timm.create_model("efficientnetv2_rw_s", pretrained=False, in_chans=1)
    print(model)
    print(torchinfo.summary(model, input_size=(8, 1, 1536, 1410), device="cpu"))
    model.blocks[5][-1].conv_pwl.ou
    print("a")
