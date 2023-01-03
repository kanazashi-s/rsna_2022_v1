from single.last_2_images.lit_module import LitModel
from single.last_2_images.config import Last2ImagesCFG
import timm
import torch


if __name__ == "__main__":
    Last2ImagesCFG.model_name = "mvitv2_large"
    model = LitModel()
    print(model)
