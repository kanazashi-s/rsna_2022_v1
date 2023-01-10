from pathlib import Path
import pandas as pd
from cfg.general import GeneralCFG


def train():
    input_path = GeneralCFG.raw_data_dir
    return pd.read_csv(input_path / "train.csv")


def test():
    input_path = GeneralCFG.raw_data_dir
    return pd.read_csv(input_path / "test.csv")


def sample_submission():
    input_path = GeneralCFG.raw_data_dir
    return pd.read_csv(input_path / "sample_submission.csv")
