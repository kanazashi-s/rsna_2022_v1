from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG


def prepare_input(input_img_name, transform, is_inference=False):
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    image = read_image(str(img_dir / input_img_name), mode=ImageReadMode.GRAY).float()
    image = transform(image)
    return image


def modify_labels(label: int, row: pd.Series) -> float:
    """
    Modify labels to be more smooth
    """
    if label == 1:
        ret_val = 1
        if row["invasive"] == 1:
            ret_val -= 0.02
        else:
            ret_val -= 0.05
        return ret_val
    else:
        ret_val = 0
        if row["BIRADS"] in [1, 2]:
            # 良性もしくは異常所見なしと、マンモグラムで判断できるとのことなので、ほぼ0
            ret_val += 0.02
            return ret_val
        else:
            ret_val += 0.05
            if row["difficult_negative_case"] == 1:
                ret_val += 0.02
            if row["biopsy"] == 1:
                ret_val += 0.03
            if row["density"] == "D":
                ret_val += 0.03
            return ret_val


class TrainDataset(Dataset):
    def __init__(self, input_df, transform, is_validation=False):
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.transform = transform
        self.is_validation = is_validation

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx]
        if GeneralCFG.train_image_dir.stem in ["lossless", "1536_ker_png"]:
            inputs = prepare_input(
                Path(str(row["machine_id"]), str(row["patient_id"]), str(row["image_id"])).with_suffix(".png"),
                self.transform,
                is_inference=False
            )
        else:
            inputs = prepare_input(
                row["image_filename"],
                self.transform,
                is_inference=False
            )
        label = torch.tensor(row[GeneralCFG.target_col]).float().unsqueeze(0)

        # if not self.is_validation:
        #     label = modify_labels(label, row)

        return inputs, label

    def get_labels(self):
        return self.input_df[GeneralCFG.target_col].values


class TestDataset(Dataset):
    def __init__(self, input_df, transform, is_inference=False):  # when is_inference=True, it is executed in kaggle notebook.
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.is_inference = is_inference
        self.transform = transform

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx]
        if GeneralCFG.train_image_dir.stem in ["lossless", "1536_ker_png"]:
            inputs = prepare_input(
                Path(str(row["machine_id"]), str(row["patient_id"]), str(row["image_id"])).with_suffix(".png"),
                self.transform,
                is_inference=self.is_inference
            )
        else:
            inputs = prepare_input(
                row["image_filename"],
                self.transform,
                is_inference=self.is_inference
            )
        return inputs
