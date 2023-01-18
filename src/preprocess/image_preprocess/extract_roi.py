# https://www.kaggle.com/code/riepom/preprocessed-dataset512/notebook

import os
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from cfg.general import GeneralCFG


BLUR = 21
CANNY_THRESH_1 = 1
CANNY_THRESH_2 = 10
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format
kernel = np.ones((5, 5), np.uint8)


def get_masked_image(path: Path):
    gray = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, kernel)
    edges = cv2.erode(edges, kernel)

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    try:
        max_contour = contour_info[0]
        # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        # mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = mask  # Create 3-channel alpha mask
        mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
        gray = gray.astype('float32') / 255.0  # for easy blending

        masked = mask_stack * gray  # Blend
        masked = (masked * 255).astype('uint8')

        return masked

    except IndexError:
        return gray


def process_train_images(image_size):
    input_path = GeneralCFG.raw_data_dir
    train_df = pd.read_csv(input_path / "train.csv")
    train_df["image_name"] = train_df["patient_id"].astype(str) + '_' + train_df['image_id'].astype(str) + '.png'
    processed_image_dir = GeneralCFG.png_data_dir / f"roi_extracted_{image_size}"
    shutil.rmtree(processed_image_dir, ignore_errors=True)
    processed_image_dir.mkdir(exist_ok=True, parents=True)

    for image_name in tqdm(train_df["image_name"].values):
        image_path = GeneralCFG.train_image_dir / image_name
        masked_image = get_masked_image(image_path)
        im_L = masked_image[:, :image_size//2]
        im_R = masked_image[:, image_size//2:]

        nnz_L = np.sum(im_L)
        nnz_R = np.sum(im_R)
        if nnz_L < nnz_R:
            masked_image = np.fliplr(masked_image)

        masked_image = cv2.resize(masked_image, dsize=(image_size, image_size))
        cv2.imwrite(str(processed_image_dir / image_name), masked_image)


def process_test_images(image_size):
    input_path = GeneralCFG.raw_data_dir
    test_df = pd.read_csv(input_path / "test.csv")
    test_df["image_name"] = test_df["patient_id"].astype(str) + '_' + test_df['image_id'].astype(str) + '.png'
    processed_image_dir = GeneralCFG.png_data_dir / f"test_roi_extracted_{image_size}"
    processed_image_dir.mkdir(exist_ok=True, parents=True)

    for image_name in test_df["image_name"].values:
        image_path = GeneralCFG.test_image_dir / image_name
        masked_image = get_masked_image(image_path)
        im_L = masked_image[:, :image_size//2]
        im_R = masked_image[:, image_size//2:]

        nnz_L = np.sum(im_L)
        nnz_R = np.sum(im_R)
        if nnz_L < nnz_R:
            masked_image = np.fliplr(masked_image)

        masked_image = cv2.resize(masked_image, dsize=(image_size, image_size))
        cv2.imwrite(str(processed_image_dir / image_name), masked_image)


if __name__ == "__main__":
    process_train_images(512)
    process_test_images(512)
