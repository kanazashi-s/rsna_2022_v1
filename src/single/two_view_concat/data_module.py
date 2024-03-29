from typing import Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from data import load_processed_data
from single.two_view_concat.config import TwoViewConcatCFG
from single.two_view_concat.dataset import TrainDataset, TestDataset
from single.two_view_concat.transforms import get_transforms
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
        whole_df = self._pivot_prediction_id(whole_df, stage="train")
        assert whole_df["prediction_id"].equals(load_processed_data.sample_oof(seed=self.seed)["prediction_id"])

        train_df = whole_df[whole_df["fold"] != self.fold]
        valid_df = whole_df[whole_df["fold"] == self.fold]

        test_df = load_processed_data.test(seed=self.seed)
        test_df = self._pivot_prediction_id(test_df, stage="test")
        assert test_df["prediction_id"].equals(load_processed_data.sample_submission(self.seed)["prediction_id"])

        if GeneralCFG.debug:
            train_df = train_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
            valid_df = valid_df.head(GeneralCFG.num_use_data).reset_index(drop=True)

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

    def _pivot_prediction_id(self, input_df, stage: str = "train"):
        """
        create a pivot table with prediction_id as index
        """
        output_df = input_df.copy()
        output_df["image_filename"] = output_df["patient_id"].astype(str) + "_" + output_df["image_id"].astype(str) + ".png"
        output_df = output_df.loc[output_df["view"].isin(["MLO", "CC"])].reset_index(drop=True)
        output_df = output_df.drop_duplicates(subset=['prediction_id', 'view'], keep='last').reset_index(drop=True)

        pivot_df = output_df.pivot(index='prediction_id', columns='view', values="image_filename").reset_index()
        pivot_df = pivot_df.rename(columns={"CC": "CC_image_filename", "MLO": "MLO_image_filename"})

        additional_cols = ['site_id', 'age', 'implant', 'machine_id']
        if stage == "train" or stage == "valid":
            additional_cols += ['fold', 'cancer']
        elif stage == "test":
            pass
        else:
            raise ValueError(f"Invalid stage: {stage} (must be train, valid, or test)")

        pivot_df = pivot_df.merge(
            output_df.groupby('prediction_id')[additional_cols].first().reset_index(),
            on='prediction_id',
            how='left'
        )

        if stage == "train":
            oof_df = load_processed_data.sample_oof(seed=self.seed)
            pivot_df = oof_df[["prediction_id"]].merge(
                pivot_df,
                on="prediction_id",
                how="left"
            )
        elif stage == "test":
            submission_df = load_processed_data.sample_submission(seed=self.seed)
            pivot_df = submission_df[["prediction_id"]].merge(
                pivot_df,
                on="prediction_id",
                how="left"
            )
        else:
            raise ValueError(f"Invalid stage: {stage} (must be train or test)")

        return pivot_df

    def train_dataloader(self):
        if TwoViewConcatCFG.sampler == "ImbalancedDatasetSampler":
            sampler_dict = {
                "sampler": ImbalancedDatasetSampler(
                    self.train_dataset,
                    num_samples=TwoViewConcatCFG.num_samples_per_epoch,
                ),
                "shuffle": False
            }
        elif TwoViewConcatCFG.sampler is None:
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
