from typing import Optional
from pathlib import Path
import numpy as np
import polars as pol
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from utils.model_utils.torch_samplers import AtLeastOnePositiveSampler
from data import load_processed_data_pol
from single.mean_agg.config import MeanAggCFG
from single.mean_agg.dataset import TrainDataset, TestDataset
from single.mean_agg.transforms import get_transforms
from cfg.general import GeneralCFG


class DataModule(pl.LightningDataModule):
    def __init__(self, seed: int, fold: int, batch_size: int, num_workers: int):
        super().__init__()
        self.seed = seed
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage="fit") -> None:
        whole_df = load_processed_data_pol.train(seed=self.seed)

        # polars version
        whole_df = whole_df.with_column(
            (pol.col("patient_id").cast(pol.Utf8) + "_" + pol.col("image_id").cast(pol.Utf8) + ".png").alias("image_filename")
        )
        train_df = whole_df.filter(pol.col("fold") != self.fold)
        valid_df = whole_df.filter(pol.col("fold") == self.fold)

        test_df = load_processed_data_pol.test(seed=self.seed)
        test_df = test_df.with_column(
            (pol.col("patient_id").cast(pol.Utf8) + "_" + pol.col("image_id").cast(pol.Utf8) + ".png").alias("image_filename")
        )

        transforms_no_aug = get_transforms(augment=False)
        transforms_aug = get_transforms(augment=True)
        self.train_dataset = TrainDataset(train_df, transforms_aug)
        self.valid_dataset = TrainDataset(valid_df, transforms_no_aug)
        self.test_dataset = TestDataset(test_df, transforms_no_aug, is_inference=True)
        self.val_predict_dataset = TestDataset(
            valid_df.drop(columns=[GeneralCFG.target_col, "fold"]),
            transforms_no_aug,
            is_inference=False
        )

    def train_dataloader(self):
        if MeanAggCFG.sampler == "ImbalancedDatasetSampler":
            sampler_dict = {
                "sampler": ImbalancedDatasetSampler(
                    self.train_dataset,
                    num_samples=MeanAggCFG.num_samples_per_epoch,
                ),
                "shuffle": False,
                "batch_size": self.batch_size
            }
        elif MeanAggCFG.sampler == "AtLeastOnePositiveSampler":
            sampler_dict = {
                "batch_sampler": AtLeastOnePositiveSampler(
                    indices=np.arange(len(self.train_dataset)),
                    labels=self.train_dataset.input_df[GeneralCFG.target_col].to_numpy(),
                    batch_size=self.batch_size,
                ),
                # "shuffle": False
            }
        elif MeanAggCFG.sampler is None:
            sampler_dict = {
                "shuffle": True,
                "batch_size": self.batch_size
            }
        else:
            raise NotImplementedError

        return DataLoader(
            self.train_dataset,
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
