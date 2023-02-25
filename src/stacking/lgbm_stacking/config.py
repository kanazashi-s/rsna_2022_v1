import os
from pathlib import Path
from cfg.general import GeneralCFG
from features import meta_feature, counts, model_output


class LGBMStackingCFG:
    output_dir = Path("/workspace", "output", "stacking", "lgbm_stacking")
    upload_name = "lgbm-stacking-20230212"
    model_name = "lgbm_stacking"
    use_features = [
        meta_feature.meta_feature,
        meta_feature.age_value,
        meta_feature.num_images,
        meta_feature.implant_value,
        counts.count_machine_id,
        model_output.model_output_agg,
    ]
    lgbm_params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_data_in_leaf": 300,
        "min_child_samples": 50,
        "scale_pos_weight": 1.1
        # is_unbalance=True,
    }
    early_stopping_rounds = 50
    verbose = -1


if GeneralCFG.is_kaggle:
    MeanAggCFG.uploaded_model_dir = Path("/kaggle", "input", LGBMStackingCFG.uploaded_model_dir)


if GeneralCFG.debug:
    MeanAggCFG.output_dir = Path("/workspace", "output", "stacking", "debug_lgbm_stacking")
