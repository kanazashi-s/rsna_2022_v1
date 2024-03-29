from pathlib import Path
import pandas as pd
from cfg.general import GeneralCFG


def train(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pd.read_csv(input_path / "train.csv")


def test(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pd.read_csv(input_path / "test.csv")


def sample_submission(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pd.read_csv(input_path / "sample_submission.csv")


def sample_oof(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pd.read_csv(input_path / "sample_oof.csv")
