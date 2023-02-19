from typing import Optional
from pathlib import Path
import numpy as np
import polars as pol
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from utils.model_utils.torch_samplers import OnePositiveSampler, StratifiedOnePositiveSampler, StratifiedSampler
from data import load_processed_data_pol
from single.cross_view.config import CrossViewCFG
from single.cross_view.dataset import TrainDataset, TestDataset
from single.cross_view.transforms import get_transforms
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

        if not GeneralCFG.debug:
            train_dicom_df = load_processed_data_pol.train_dicom()
            whole_df = pol.concat([whole_df, train_dicom_df], how="horizontal")
            assert whole_df.get_column("PatientID").series_equal(whole_df.get_column("patient_id").alias("PatientID"))
            assert whole_df.get_column("laterality").series_equal(whole_df.get_column("ImageLaterality").alias("laterality"))

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
        self.valid_dataset = TrainDataset(valid_df, transforms_no_aug, is_validation=True)
        self.test_dataset = TestDataset(test_df, transforms_no_aug, is_inference=True)
        self.val_predict_dataset = TestDataset(
            valid_df.drop(columns=[GeneralCFG.target_col, "fold"]),
            transforms_no_aug,
            is_inference=False
        )

    def train_dataloader(self):
        if CrossViewCFG.sampler == "ImbalancedDatasetSampler":
            sampler_dict = {
                "sampler": ImbalancedDatasetSampler(
                    self.train_dataset,
                    num_samples=CrossViewCFG.num_samples_per_epoch,
                ),
                "shuffle": False,
                "batch_size": self.batch_size
            }
        elif CrossViewCFG.sampler == "OnePositiveSampler":
            sampler_dict = {
                "batch_sampler": OnePositiveSampler(
                    indices=np.arange(len(self.train_dataset)),
                    labels=self.train_dataset.input_df[GeneralCFG.target_col].to_numpy(),
                    batch_size=self.batch_size,
                ),
                # "shuffle": False
            }
        elif CrossViewCFG.sampler == "StratifiedOnePositiveSampler":
            sampler_dict = {
                "batch_sampler": StratifiedOnePositiveSampler(
                    indices=np.arange(len(self.train_dataset)),
                    labels=self.train_dataset.input_df[GeneralCFG.target_col].to_numpy(),
                    batch_size=self.batch_size,
                    groups=self.train_dataset.input_df["Rows"].to_numpy(),
                ),
                # "shuffle": False
            }
        elif CrossViewCFG.sampler is None:
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
        if CrossViewCFG.sampler == "StratifiedOnePositiveSampler":
            sampler_dict = {
                "batch_sampler": StratifiedSampler(
                    indices=np.arange(len(self.valid_dataset)),
                    groups=self.valid_dataset.input_df["Rows"].to_numpy(),
                    batch_size=self.batch_size,
                ),
            }
        else:
            sampler_dict = {
                "shuffle": False,
                "batch_size": self.batch_size
            }
        return DataLoader(
            self.valid_dataset,
            num_workers=self.num_workers,
            **sampler_dict
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
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            shuffle=False
        )


if __name__ == "__main__":
    data_module = DataModule(seed=42, fold=0, batch_size=2, num_workers=0)
    data_module.setup()
    for inputs, labels in data_module.train_dataloader():
        print(inputs)
        print(labels)
        break
