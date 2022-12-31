from pathlib import Path


class GeneralCFG:
    data_dir = Path("/workspace", "data")
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    data_version = "vanilla"
    debug = False
    num_workers = 1
    seeds = [42, 43, 44]
    n_fold = 5
    train_fold = [0, 1, 2, 3, 4]
    num_use_data = None
    target_col = "hoge"


if GeneralCFG.debug:
    GeneralCFG.train_fold = [0, 1]
    GeneralCFG.num_use_data = 300
