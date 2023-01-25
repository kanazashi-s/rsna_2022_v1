from pathlib import Path
import polars as pol
from cfg.general import GeneralCFG


def train(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    file_name = "debug_train.csv" if GeneralCFG.debug else "train.csv"
    return pol.read_csv(input_path / file_name)


def test(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pol.read_csv(input_path / "test.csv")


def sample_submission(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    return pol.read_csv(input_path / "sample_submission.csv")


def sample_oof(seed: int):
    input_path = GeneralCFG.processed_data_dir / GeneralCFG.data_version / f"seed{seed}"
    file_name = "debug_sample_oof.csv" if GeneralCFG.debug else "sample_oof.csv"
    return pol.read_csv(input_path / file_name)

