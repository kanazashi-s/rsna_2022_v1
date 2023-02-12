from pathlib import Path
import shutil
from collections import defaultdict
import lightgbm as lgb
import polars as pol
import mlflow
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from features import build_features
from stacking.lgbm_stacking.config import LGBMStackingCFG
from stacking.lgbm_stacking.evaluate import evaluate


def train(run_name: str, seed_list=None):
    metrics_seed_dict = defaultdict(list)
    if seed_list is None:
        seed_list = GeneralCFG.seeds
        
    for seed in seed_list:
        if GeneralCFG.debug:
            LGBMStackingCFG.output_dir = LGBMStackingCFG.output_dir.with_name(f"{LGBMStackingCFG.output_dir.name}_debug")
        output_dir = LGBMStackingCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        whole_base_df = load_processed_data_pol.sample_oof(seed=seed).select(
            pol.col("prediction_id")
        )
        whole_features_df = build_features.make(
            use_features=LGBMStackingCFG.use_features,
            X_base=whole_base_df,
            seed=seed
        )

        # mlflow logger
        experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
        mlflow.set_experiment(experiment_name_prefix + f"lgbm_stacking_{LGBMStackingCFG.model_name}")

        for fold in GeneralCFG.train_fold:
            mlflow.lightgbm.autolog(log_models=False)

            model_save_name = f"best_loss_fold{fold}"

            train_features_df = whole_features_df.filter(
                pol.col("fold") != fold
            )
            valid_features_df = whole_features_df.filter(
                pol.col("fold") == fold
            )

            meta_cols = ["prediction_id", "fold"]
            train_X = train_features_df.drop(meta_cols + [GeneralCFG.target_col]).to_numpy()
            train_y = train_features_df.select(GeneralCFG.target_col).to_numpy()
            valid_X = valid_features_df.drop(meta_cols + [GeneralCFG.target_col]).to_numpy()
            valid_y = valid_features_df.select(GeneralCFG.target_col).to_numpy()

            model = lgb.train(
                params=LGBMStackingCFG.lgbm_params,
                train_set=lgb.Dataset(train_X, train_y),
                valid_sets=[lgb.Dataset(valid_X, valid_y)],
                early_stopping_rounds=LGBMStackingCFG.early_stopping_rounds,
                verbose_eval=LGBMStackingCFG.verbose,
            )
            model.save_model(str(output_dir / f"{model_save_name}.txt"))

        whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed)
        log_all_metrics(whole_metrics, metrics_by_folds, metrics_each_fold)

        metrics_seed_dict["whole_best_pfbeta"].append(whole_metrics["best_pfbeta"])
        metrics_seed_dict["whole_auc_roc"].append(whole_metrics["auc_roc"])
        metrics_seed_dict["whole_auc_pr"].append(whole_metrics["auc_pr"])
        metrics_seed_dict["by_folds_best_pfbeta"].append(metrics_by_folds["pfbeta"])
        metrics_seed_dict["by_folds_auc_roc"].append(metrics_by_folds["auc_roc"])
        metrics_seed_dict["by_folds_auc_pr"].append(metrics_by_folds["auc_pr"])

    # metrics_seed_dict のすべての要素に対し、平均を計算し、ロギングする
    metrics_seed_mean_dict = {}
    for key, value in metrics_seed_dict.items():
        metrics_seed_mean_dict[f"seed_mean_{key}"] = np.mean(value)
    mlflow_logger.log_metrics(metrics_seed_mean_dict)

    create_dataset_metadata(
        model_name=LGBMStackingCFG.upload_name,
        model_path=LGBMStackingCFG.output_dir,
    )

    return metrics_seed_mean_dict


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
    }
    return param_dict


def log_all_metrics(whole_metrics, metrics_by_folds, metrics_each_fold):
    num_metrics_dict = {
        "whole_pfbeta": whole_metrics["best_pfbeta"],
        "whole_best_threshold": whole_metrics["best_thresh_pfbeta"],
        "whole_auc_roc": whole_metrics["auc_roc"],
        "whole_auc_pr": whole_metrics["auc_pr"],
        "whole_TP": whole_metrics["TP"],
        "whole_FP": whole_metrics["FP"],
        "whole_FN": whole_metrics["FN"],
        "whole_TN": whole_metrics["TN"],
        "by_fold_pfbeta": metrics_by_folds["pfbeta"],
        "by_fold_auc_roc": metrics_by_folds["auc_roc"],
        "by_fold_auc_pr": metrics_by_folds["auc_pr"],
        "by_fold_TP": metrics_by_folds["TP"],
        "by_fold_FP": metrics_by_folds["FP"],
        "by_fold_FN": metrics_by_folds["FN"],
        "by_fold_TN": metrics_by_folds["TN"],
    }
    for i, metrics_one_fold in enumerate(metrics_each_fold):
        num_metrics_dict[f"fold{i}_pfbeta"] = metrics_one_fold["best_pfbeta"]
        num_metrics_dict[f"fold{i}_best_threshold"] = metrics_one_fold["best_thresh_pfbeta"]
        num_metrics_dict[f"fold{i}_auc_roc"] = metrics_one_fold["auc_roc"]
        num_metrics_dict[f"fold{i}_auc_pr"] = metrics_one_fold["auc_pr"]
        num_metrics_dict[f"fold{i}_TP"] = metrics_one_fold["TP"]
        num_metrics_dict[f"fold{i}_FP"] = metrics_one_fold["FP"]
        num_metrics_dict[f"fold{i}_FN"] = metrics_one_fold["FN"]
        num_metrics_dict[f"fold{i}_TN"] = metrics_one_fold["TN"]

    mlflow.log_metrics(num_metrics_dict)

    # log fig metrics
    run_id = mlflow_logger.run_id
    for key, value in whole_metrics.items():
        if key.endswith("curve"):
            mlflow_logger.experiment.log_figure(run_id, value, f"whole_{key}.png")
    for i, metrics_one_fold in enumerate(metrics_each_fold):
        for key, value in metrics_one_fold.items():
            if key.endswith("curve"):
                mlflow_logger.experiment.log_figure(run_id, value, f"each_fold_{i}_{key}.png")


if __name__ == "__main__":
    LGBMStackingCFG.output_dir = Path("/workspace", "output", "stacking", "lgbm_stacking_baseline")
    train(run_name="lgbm_stacking_baseline", seed_list=[42])


