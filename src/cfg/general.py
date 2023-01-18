from pathlib import Path
import os


class GeneralCFG:
    is_kaggle = bool(int(os.environ["IS_KAGGLE_ENVIRONMENT"]))
    data_dir = Path("/kaggle", "input", "rsna-breast-cancer-detection") if is_kaggle else Path("/workspace", "data")
    raw_data_dir = Path("/kaggle", "input", "rsna-breast-cancer-detection") if is_kaggle else data_dir / "raw"
    processed_data_dir = Path("/kaggle", "working", "processed") if is_kaggle else data_dir / "processed"
    png_data_dir = data_dir / "png_converted"
    train_image_dir = png_data_dir / "test_png_512"
    test_image_dir = Path("/kaggle", "working", "test_png_512") if is_kaggle else png_data_dir / "test_png_512"
    image_size = 512
    data_version = "vanilla"
    debug = False
    num_workers = 2
    seeds = [42, 43, 44]
    n_fold = 5
    train_fold = [0, 1, 2, 3, 4]
    num_use_data = None
    target_col = "cancer"


if GeneralCFG.is_kaggle:
    GeneralCFG.j2k_dir = Path("/kaggle", "working", "j2k_dir")


if GeneralCFG.debug:
    GeneralCFG.test_image_dir = Path("/kaggle", "working", "train_png_512")
    GeneralCFG.train_fold = [0, 1]
    GeneralCFG.num_use_data = 300
