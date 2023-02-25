import os
from collections import OrderedDict
import timm
import torch
import torch.nn as nn
import torchinfo


if __name__ == "__main__":
    print(timm.list_models("*maxxvit*", pretrained=True))
    compare_model = timm.create_model("maxxvit_rmlp_nano_rw_256", pretrained=False, in_chans=1)
    effnetv2_s = timm.create_model("efficientnetv2_s", pretrained=False, in_chans=1)

    input_shape = (8, 1, 1536, 1410)
    torchinfo.summary(compare_model, input_size=input_shape)
    torchinfo.summary(effnetv2_s, input_size=input_shape)

    print("a")