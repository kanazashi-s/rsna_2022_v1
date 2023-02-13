import os
import timm
import torchinfo


if __name__ == "__main__":
    print(timm.list_models("*vit*", pretrained=True))
    model = timm.create_model("efficientnetv2_rw_m", pretrained=False, in_chans=1)
    print(model)
    print(torchinfo.summary(model, input_size=(8, 1, 1536, 1536), device="cpu"))
