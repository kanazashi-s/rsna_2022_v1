from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from cfg.general import GeneralCFG
from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG


def prepare_input(input_img_name, is_inference=False):
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    image = read_image(str(img_dir / input_img_name), mode=ImageReadMode.GRAY).float()
    return image


class ExtractDataset(Dataset):
    def __init__(self, input_df, is_inference=False):
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.is_inference = is_inference

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx]
        inputs = prepare_input(
            Path(str(row["machine_id"]), str(row["patient_id"]), str(row["image_id"])).with_suffix(".png"),
            is_inference=self.is_inference
        )
        return inputs


def get_extract_dataloader(input_df, batch_size=32, num_workers=4):
    is_inference = GeneralCFG.is_kaggle
    dataset = ExtractDataset(input_df, is_inference=is_inference)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


if __name__ == "__main__":
    from data import load_processed_data_pol
    train_df = load_processed_data_pol.train(42)
    dataloader = get_extract_dataloader(train_df)
    for inputs in dataloader:
        print(inputs.shape)
        break