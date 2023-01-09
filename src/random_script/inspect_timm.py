import os
import timm
import torchinfo


if __name__ == "__main__":
    # print(timm.list_models("*efficientnet*", pretrained=True))
    # model = timm.create_model("efficientnet_b0", pretrained=False, in_chans=1)
    # print(model)
    # print(torchinfo.summary(model, input_size=(8, 1, 1536, 1536)))

    print(os.listdir("/workspace/data_hdd"))
