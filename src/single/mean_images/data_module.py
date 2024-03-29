from typing import Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from data import load_processed_data
from single.mean_images.config import MeanImagesCFG
from single.mean_images.dataset import TrainDataset, TestDataset
from single.mean_images.transforms import get_transforms
from cfg.general import GeneralCFG


class DataModule(pl.LightningDataModule):
    def __init__(self, seed: int, fold: int, batch_size: int, num_workers: int):
        super().__init__()
        self.seed = seed
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage="fit") -> None:
        whole_df = load_processed_data.train(seed=self.seed)
        whole_df["image_filename"] = whole_df["patient_id"].astype(str) + "_" + whole_df["image_id"].astype(str) + ".png"

        self.train_df = whole_df[whole_df["fold"] != self.fold]
        self.valid_df = whole_df[whole_df["fold"] == self.fold]

        self.test_df = load_processed_data.test(seed=self.seed)
        self.test_df["image_filename"] = self.test_df["patient_id"].astype(str) + "_" + self.test_df["image_id"].astype(str) + ".png"

        if GeneralCFG.debug:
            self.train_df = self.train_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
            self.valid_df = self.valid_df.head(GeneralCFG.num_use_data).reset_index(drop=True)

        transforms_no_aug = get_transforms(augment=False)
        transforms_aug = get_transforms(augment=True)
        self.train_dataset = TrainDataset(self.train_df, transforms_aug)
        self.valid_dataset = TrainDataset(self.valid_df, transforms_no_aug)
        self.test_dataset = TestDataset(self.test_df, transforms_no_aug, is_inference=True)
        self.val_predict_dataset = TestDataset(
            self.valid_df.drop(columns=[GeneralCFG.target_col, "fold"]),
            transforms_no_aug,
            is_inference=False
        )

    def train_dataloader(self):
        if MeanImagesCFG.sampler == "ImbalancedDatasetSampler":
            sampler_dict = {
                "sampler": ImbalancedDatasetSampler(self.train_dataset),
                "shuffle": False
            }
        elif MeanImagesCFG.sampler is None:
            sampler_dict = {
                "shuffle": True
            }
        else:
            raise NotImplementedError

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **sampler_dict
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


if __name__ == "__main__":
    data_module = DataModule(seed=42, fold=0, batch_size=2, num_workers=1)
    data_module.setup()
    for inputs, labels in data_module.train_dataloader():
        print(inputs)
        print(labels)
        break
