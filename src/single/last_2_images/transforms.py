import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import torch
import torch.nn as nn


def get_transforms(augment=True, visualize=False):
    transforms_list = [
        # T.Resize((320, 320)),
        # T.Normalize(mean=0.449, std=0.226),
    ]

    if augment:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(90),
            # T.RandomRotation(180),
            # T.RandomRotation(270),
            # T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    return T.Compose(transforms_list)
