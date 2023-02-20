from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from cfg.general import GeneralCFG
from single.cross_view.config import CrossViewCFG


def name_to_image(input_img_name, transform, is_inference=False):
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    image = read_image(str(img_dir / input_img_name), mode=ImageReadMode.GRAY).float()
    image = transform(image)
    return image


class TrainDataset(Dataset):
    def __init__(self, input_df, transform, is_validation=False):
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.transform = transform
        self.is_validation = is_validation

        if GeneralCFG.train_image_dir.stem in ["lossless", "1536_ker_png"]:
            self.input_df["image_filename"] = self.input_df["machine_id"].astype(str) + "/" \
                                              + self.input_df["patient_id"].astype(str) + "/" \
                                              + self.input_df["image_id"].astype(str) + ".png"

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, patient_id):
        rows = self.input_df.iloc[patient_id]
        lcc_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "CC")]
        rcc_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "CC")]
        lmlo_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "MLO")]
        rmlo_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "MLO")]

        lcc_image_names = lcc_rows["image_filename"].values
        rcc_image_names = rcc_rows["image_filename"].values
        lmlo_image_names = lmlo_rows["image_filename"].values
        rmlo_image_names = rmlo_rows["image_filename"].values

        lcc_images = torch.stack([name_to_image(image_name, self.transform) for image_name in lcc_image_names])
        rcc_images = torch.stack([name_to_image(image_name, self.transform) for image_name in rcc_image_names])
        lmlo_images = torch.stack([name_to_image(image_name, self.transform) for image_name in lmlo_image_names])
        rmlo_images = torch.stack([name_to_image(image_name, self.transform) for image_name in rmlo_image_names])

        inputs = [lcc_images, rcc_images, lmlo_images, rmlo_images]

        l_label = int(rows.loc[rows["laterality"] == "L"]["cancer"].max())
        r_label = int(rows.loc[rows["laterality"] == "R"]["cancer"].max())

        labels = torch.stack([l_label, r_label])

        return inputs, labels


class TestDataset(Dataset):
    def __init__(self, input_df, transform, is_inference=False):  # when is_inference=True, it is executed in kaggle notebook.
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.is_inference = is_inference
        self.transform = transform

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        rows = self.input_df.iloc[idx]
        lcc_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "CC")]
        rcc_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "CC")]
        lmlo_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "MLO")]
        rmlo_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "MLO")]

        lcc_image_names = lcc_rows["image_filename"].values
        rcc_image_names = rcc_rows["image_filename"].values
        lmlo_image_names = lmlo_rows["image_filename"].values
        rmlo_image_names = rmlo_rows["image_filename"].values

        lcc_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in lcc_image_names])
        rcc_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in rcc_image_names])
        lmlo_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in lmlo_image_names])
        rmlo_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in rmlo_image_names])

        inputs = [lcc_images, rcc_images, lmlo_images, rmlo_images]

        return inputs
