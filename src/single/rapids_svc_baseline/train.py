from pathlib import Path
import shutil
import pickle
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import polars as pol
import mlflow
from cuml.svm import SVC
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from features import build_features
from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
from single.rapids_svc_baseline.feature_extract import effnet_v2_m
from single.rapids_svc_baseline.feature_extract import pretrain_effnet_v2_s
from single.rapids_svc_baseline.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def train(seed_list=None):
    metrics_seed_dict = defaultdict(list)
    if seed_list is None:
        seed_list = GeneralCFG.seeds

    for seed in seed_list:
        shutil.rmtree(RapidsSvcBaselineCFG.output_dir, ignore_errors=True)
        RapidsSvcBaselineCFG.output_dir.mkdir(parents=True, exist_ok=True)

        whole_base_df = load_processed_data_pol.train(seed=42)

        effnet_v2_m_features = effnet_v2_m.extract(whole_base_df)
        pretrain_effnet_v2_s_features = pretrain_effnet_v2_s.extract(whole_base_df)

        whole_df = whole_base_df.join(
            effnet_v2_m_features, on="image_id", how="left"
        ).join(
            pretrain_effnet_v2_s_features, on="image_id", how="left"
        )

        X = whole_df.drop(["prediction_id", "image_id", "fold", GeneralCFG.target_col]).to_numpy()
        y = whole_df.get_column(GeneralCFG.target_col).to_numpy()
        folds = whole_df.get_column("fold").to_numpy()

        for fold in GeneralCFG.train_fold:
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + f"rapids_svc_baseline",
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            X_train = X[folds != fold]
            y_train = y[folds != fold]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            pickle.dump(scaler, open(RapidsSvcBaselineCFG.output_dir / f"scaler_fold{fold}.pkl", "wb"))

            model = SVC(
                C=RapidsSvcBaselineCFG.C,
            )
            model.fit(X_train, y_train)
            pickle.dump(model, open(RapidsSvcBaselineCFG.output_dir / f"model_fold{fold}.pkl", "wb"))

        whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed, device_idx)
        log_all_metrics(mlflow_logger, whole_metrics, metrics_by_folds, metrics_each_fold)

        create_dataset_metadata(
            model_name=f"{RapidsSvcBaselineCFG.upload_name}-seed{seed}",
            model_path=output_dir,
        )

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
        model_name=RapidsSvcBaselineCFG.upload_name,
        model_path=RapidsSvcBaselineCFG.output_dir,
    )

    return metrics_seed_mean_dict