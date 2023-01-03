import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from cfg.general import GeneralCFG
from single.last_2_images.config import Last2ImagesCFG


def prepare_input(input_img_names, transform, is_inference=False):
    mlo_image_filename, cc_image_filename = input_img_names
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    mlo_image = read_image(str(img_dir / mlo_image_filename), mode=ImageReadMode.GRAY).float()
    cc_image = read_image(str(img_dir / cc_image_filename), mode=ImageReadMode.GRAY).float()

    # mlo_image = transform(image=np.array(mlo_image))["image"]
    # cc_image = transform(image=np.array(cc_image))["image"]
    mlo_image = transform(mlo_image)
    cc_image = transform(cc_image)

    return mlo_image, cc_image


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
            row[["CC_image_filename", "MLO_image_filename"]].values,
            self.transform,
            is_inference=False
        )
        label = torch.tensor(row[GeneralCFG.target_col]).float().unsqueeze(0)
        return inputs, label

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
            row[["CC_image_filename", "MLO_image_filename"]].values,
            self.transform,
            is_inference=self.is_inference
        )
        return inputs
