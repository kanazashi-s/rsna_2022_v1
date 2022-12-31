from typing import Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import load_processed_data
from data.dataset import TrainDataset, TestDataset
from config.general import GeneralCFG


class DataModule(pl.LightningDataModule):
    def __init__(self, seed: int, fold: int, batch_size: int, num_workers: int):
        super().__init__()
        self.seed = seed
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self) -> None:
        whole_df = load_processed_data.train(seed=self.seed)

        train_df = whole_df[whole_df["fold"] != self.fold]
        train_df = train_df.drop(["fold", "id_col"], axis=1)
        valid_df = whole_df[whole_df["fold"] == self.fold]
        valid_df = valid_df.drop(["fold", "id_col"], axis=1)

        test_df = load_processed_data.test(seed=self.seed)
        test_df = test_df.drop(["id_col"], axis=1)

        if GeneralCFG.debug:
            train_df = train_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
            valid_df = valid_df.head(GeneralCFG.num_use_data).reset_index(drop=True)

        self.train_dataset = TrainDataset(train_df)
        self.valid_dataset = TrainDataset(valid_df)
        self.test_dataset = TestDataset(test_df)
        self.val_predict_dataset = TestDataset(valid_df[["full_text"]])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
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
    data_module = DataModule(seed=42, fold=0, batch_size=1, num_workers=1)
    data_module.setup()
    for inputs, labels in data_module.train_dataloader():
        print(inputs)
        print(labels)
        break