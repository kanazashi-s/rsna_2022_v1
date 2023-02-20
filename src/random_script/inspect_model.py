import timm
import torch
import torch.nn as nn


if __name__ == "__main__":
    sample_input = torch.rand(8, 16, 100, 80)

    max_pool_3d = nn.MaxPool3d(kernel_size=(16, 1, 1))(sample_input)
    max_pool_2d = nn.MaxPool2d(kernel_size=(sample_input.size()[2:]))(sample_input)


