import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from cfg.general import GeneralCFG
from single.mean_images.config import MeanImagesCFG


def prepare_input(input_img_name, transform, is_inference=False):
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    input_img = read_image(str(img_dir / input_img_name), mode=ImageReadMode.GRAY).float()
    input_img = transform(input_img)

    return input_img


class TrainDataset(Dataset):
    def __init__(self, input_df, transform):
        super().__init__()
        self.input_df = input_df
        self.transform = transform

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx]
        inputs = prepare_input(
            row["image_filename"],
            self.transform,
            is_inference=False
        )
        label = torch.tensor(row[GeneralCFG.target_col]).float().unsqueeze(0)
        # prediction_id = row["prediction_id"]
        return inputs, label,   # prediction_id

    def get_labels(self):
        return self.input_df[GeneralCFG.target_col].values


class TestDataset(Dataset):
    def __init__(self, input_df, transform, is_inference=False):  # when is_inference=True, it is executed in kaggle notebook.
        super().__init__()
        self.input_df = input_df
        self.is_inference = is_inference
        self.transform = transform

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx]
        inputs = prepare_input(
            row["image_filename"].values,
            self.transform,
            is_inference=self.is_inference
        )
        # prediction_id = row["prediction_id"]
        return inputs,   # prediction_id
