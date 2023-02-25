import json
from pathlib import Path
import shutil
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import polars as pol
import optuna
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from features import build_features
from stacking.lgbm_stacking.config import LGBMStackingCFG
from stacking.lgbm_stacking.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def objective(trial, seed_list=None):
    params = {
        "objective": "binary",
        "eval_metric": "average_precision",
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 10 - 1),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "verbose": -1,
    }

    metrics_seed_dict = defaultdict(list)
    if seed_list is None:
        seed_list = [42]

    for seed in seed_list:
        output_dir = LGBMStackingCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        whole_base_df = load_processed_data_pol.sample_oof(seed=seed).select(
            pol.col("prediction_id")
        )
        whole_features_df = build_features.make(
            use_features=LGBMStackingCFG.use_features,
            X_base=whole_base_df,
            seed=seed,
        )

        for fold in GeneralCFG.train_fold:
            train_features_df = whole_features_df.filter(
                pol.col("fold") != fold
            )
            valid_features_df = whole_features_df.filter(
                pol.col("fold") == fold
            )

            meta_cols = ["prediction_id", "fold"]
            train_X = train_features_df.drop(meta_cols + [GeneralCFG.target_col]).to_pandas()
            train_y = train_features_df.select(GeneralCFG.target_col).to_pandas()
            valid_X = valid_features_df.drop(meta_cols + [GeneralCFG.target_col]).to_pandas()
            valid_y = valid_features_df.select(GeneralCFG.target_col).to_pandas()

            model = lgb.train(
                params=params,
                train_set=lgb.Dataset(train_X, train_y),
                valid_sets=[lgb.Dataset(valid_X, valid_y)],
                num_boost_round=10000,
                early_stopping_rounds=100,
            )

            model.save_model(str(output_dir / f"best_loss_fold{fold}.txt"))

        whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed)
        metrics_seed_dict["whole_best_pfbeta"].append(whole_metrics["best_pfbeta"])
        metrics_seed_dict["whole_auc_roc"].append(whole_metrics["auc_roc"])
        metrics_seed_dict["whole_auc_pr"].append(whole_metrics["auc_pr"])
        metrics_seed_dict["by_folds_best_pfbeta"].append(metrics_by_folds["pfbeta"])
        metrics_seed_dict["by_folds_auc_roc"].append(metrics_by_folds["auc_roc"])
        metrics_seed_dict["by_folds_auc_pr"].append(metrics_by_folds["auc_pr"])

    metrics_seed_mean_dict = {}
    for key, value in metrics_seed_dict.items():
        metrics_seed_mean_dict[f"seed_mean_{key}"] = np.mean(value)

    return metrics_seed_mean_dict["seed_mean_whole_auc_pr"]


def main():
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_stacking",
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # save best params
    best_params = trial.params
    best_params["objective"] = "binary"
    best_params["eval_metric"] = "average_precision"
    best_params["seed"] = 42
    best_params["num_boost_round"] = 10000
    best_params["early_stopping_rounds"] = 100
    best_params["verbose_eval"] = 100
    best_params["use_features"] = LGBMStackingCFG.use_features
    best_params["n_trials"] = 100

    with open(LGBMStackingCFG.output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)


if __name__ == "__main__":
    main()
