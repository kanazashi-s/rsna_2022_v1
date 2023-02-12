from pathlib import Path
import lightgbm as lgb
import polars as pol
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from features import build_features
from stacking.lgbm_stacking.config import LGBMStackingCFG
from metrics import calc_oof_score_pol


def evaluate(seed):
    oof_df = load_processed_data_pol.sample_oof(seed=seed).with_column(
        pol.lit(0).alias("prediction")
    )
    whole_base_df = oof_df.select(
        pol.col("prediction_id")
    )
    whole_features_df = build_features.make(use_features=LGBMStackingCFG.use_features, X_base=whole_base_df, seed=seed)

    for fold in GeneralCFG.train_fold:
        input_dir = LGBMStackingCFG.output_dir / f"seed{seed}"
        meta_cols = ["prediction_id", "fold"]
        valid_features_df = whole_features_df.filter(
            pol.col("fold") == fold
        )
        valid_X = valid_features_df.drop(meta_cols + GeneralCFG.target_col).to_numpy()

        model = lgb.Booster(model_file=f"{input_dir}/best_loss_fold{fold}.txt")
        fold_i_val_pred = model.predict(valid_X)

        oof_df = oof_df.join(
            oof_df.filter(
                pol.col("fold") == fold
            ).select(
                pol.col("prediction_id")
            ).with_column(
                pol.lit(fold_i_val_pred.reshape(-1)).alias("prediction_i")
            ),
            on="prediction_id",
            how="left"
        ).select([
            pol.col("prediction_id"),
            pol.col("fold"),
            (pol.col("prediction") + pol.col("prediction_i")).alias("prediction"),
        ])

    oof_df.write_csv(LGBMStackingCFG.output_dir / f"oof.csv")
    whole_metrics, metrics_by_folds, metrics_each_fold = calc_oof_score_pol.calc(
        oof_df, is_debug=debug, seed=seed, is_sigmoid=True
    )
    return whole_metrics, metrics_by_folds, metrics_each_fold


if __name__ == "__main__":
    LGBMStackingCFG.output_dir = Path("/workspace", "output", "stacking", "lgbm_stacking_baseline")
    whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed=42)
    print(whole_metrics)
