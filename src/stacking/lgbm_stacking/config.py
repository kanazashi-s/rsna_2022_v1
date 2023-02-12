import os
from pathlib import Path
from cfg.general import GeneralCFG


class LGBMStackingCFG:
    output_dir = Path("/workspace", "output", "stacking", "lgbm_stacking")
    upload_name = "lgbm-stacking-20230212"
    lgbm_params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "max_bin": 255,
    }
    eval_metric = "average_precision"
    early_stopping_rounds = 50
    verbose = 1


if GeneralCFG.is_kaggle:
    MeanAggCFG.uploaded_model_dir = Path("/kaggle", "input", LGBMStackingCFG.uploaded_model_dir)


if GeneralCFG.debug:
    MeanAggCFG.output_dir = Path("/workspace", "output", "stacking", "debug_lgbm_stacking")
