from pathlib import Path
import os


class GeneralCFG:
    is_kaggle = bool(int(os.environ["IS_KAGGLE_ENVIRONMENT"]))
    data_dir = Path("/kaggle", "input", "rsna-breast-cancer-detection") if is_kaggle else Path("/workspace", "data")
    raw_data_dir = Path("/kaggle", "input", "rsna-breast-cancer-detection") if is_kaggle else data_dir / "raw"
    processed_data_dir = Path("/kaggle", "working", "processed") if is_kaggle else data_dir / "processed"
    png_data_dir = data_dir / "png_converted"
    train_image_dir = png_data_dir / "1536_ker_png"
    test_image_dir = Path("/kaggle", "working", "1536_ker_png_test") if is_kaggle else png_data_dir / "1536_ker_png_test"
    image_size = 1536
    data_version = "vanilla"
    debug = os.environ.get('DEBUG_FLG', '0') == '1'
    num_workers = 0
    seeds = [42]
    n_fold = 5
    train_fold = [0, 1, 2, 3, 4]
    num_use_data = None
    target_col = "cancer"


if GeneralCFG.is_kaggle:
    GeneralCFG.j2k_dir = Path("/kaggle", "working", "j2k_dir")


if GeneralCFG.debug:
    GeneralCFG.test_image_dir = Path("/kaggle", "working", "1536_ker_png")
    GeneralCFG.train_fold = [0, 1]
    GeneralCFG.num_use_data = 300
