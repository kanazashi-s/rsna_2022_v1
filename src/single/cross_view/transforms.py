import math
import numpy as np
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
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    return T.Compose(transforms_list)


# #geometric
def affine_param_to_matrix(
    degree=10,
    scale=0.1,
    translate=(10,10),
    shear=(10,10),
):
    #h,w = image_shape
    #https://stackoverflow.com/questions/61242154/the-third-column-of-cv2-getrotationmatrix2d
    rotate = cv2.getRotationMatrix2D(angle=degree, center=(0, 0), scale=scale)

    # Shear
    shear_x = math.tan(shear[0] * math.pi / 180)
    shear_y = math.tan(shear[1] * math.pi / 180)

    matrix = np.ones([2, 3])
    matrix[0] = rotate[0] + shear_y * rotate[1]
    matrix[1] = rotate[1] + shear_x * rotate[0]
    matrix[0, 2] = translate[0]
    matrix[1, 2] = translate[1]
    return matrix


def get_random_affine_matrix(
    image_shape=(1536, 1410),
    degree=15,
    translate=0.,
    scale=0.2,
    shear=10,
):
    h, w = image_shape
    degree = np.random.uniform(-degree, degree)
    scale = np.random.uniform(-scale, scale) + 1
    translate_x, translate_y = np.random.uniform(-translate, translate, 2) * [w, h]
    shear_x, shear_y = np.random.uniform(-shear, shear, 2)

    matrix = affine_param_to_matrix(
        degree,
        scale,
        (translate_x, translate_y),
        (shear_x, shear_y),
    )

    return matrix


def get_random_affine_params():
    return {
        "angle": np.random.uniform(-10, 10),
        "translate": np.random.uniform(-0.1, 0.1, 2),
        "scale": np.random.uniform(-0.1, 0.1) + 1,
        "shear": np.random.uniform(-10, 10, 2),
    }