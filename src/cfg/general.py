from pathlib import Path


class GeneralCFG:
    is_kaggle = False
    data_dir = Path("/workspace", "data")
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    png_data_dir = data_dir / "png_converted"
    train_image_dir = png_data_dir / "theo_512"
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
    GeneralCFG.data_dir = Path("/kaggle", "input", "rsna-breast-cancer-detection")
    GeneralCFG.test_image_dir = Path("/kaggle", "working", "converted_test_images")
    GeneralCFG.j2k_dir = Path("/kaggle", "working", "j2k_dir")


if GeneralCFG.debug:
    GeneralCFG.test_image_dir = Path("/kaggle", "working", "train_png_512")
    GeneralCFG.train_fold = [0, 1]
    GeneralCFG.num_use_data = 300
