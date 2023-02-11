import os
from pathlib import Path
from cfg.general import GeneralCFG


class LGBMStackingCFG:
    output_dir = Path("/workspace", "output", "stacking", "lgbm_stacking")
    upload_name = "lgbm_stacking_20230212"
    lgbm_params = {
        "learning_rate": 0.5,
    }

    monitor_metric = "auc_pr"
    monitor_mode = "max"
    uploaded_model_dir = output_dir
    val_check_per_epoch = 2


if GeneralCFG.is_kaggle:
    MeanAggCFG.uploaded_model_dir = Path("/kaggle", "input", LGBMStackingCFG.uploaded_model_dir)
    MeanAggCFG.batch_size = 4


if GeneralCFG.debug:
    MeanAggCFG.epochs = 2
    MeanAggCFG.output_dir = Path("/workspace", "output", "stacking", "debug_lgbm_stacking")
