from pathlib import Path
import numpy as np
import polars as pol
import torch
import pytorch_lightning as pl
import ttach as tta
from cfg.general import GeneralCFG
from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
from data import load_processed_data_pol
from metrics import calc_oof_score_pol


def evaluate(seed, device_idx=0):
    whole_base_df = load_processed_data_pol.train(seed=42)

    effnet_v2_m_features = effnet_v2_m.extract(whole_base_df)
    pretrain_effnet_v2_s_features = pretrain_effnet_v2_s.extract(whole_base_df)

    whole_df = whole_base_df.join(
        effnet_v2_m_features, on="image_id", how="left"
    ).join(
        pretrain_effnet_v2_s_features, on="image_id", how="left"
    )
    train_df = whole_df

    X = whole_df.drop(["prediction_id", "image_id", "fold", GeneralCFG.target_col]).to_numpy()
    y = whole_df.get_column(GeneralCFG.target_col).to_numpy()
    folds = whole_df.get_column("fold").to_numpy()
    oof_df = load_processed_data_pol.sample_oof(seed=seed)

    for fold in GeneralCFG.train_fold:
        X_valid = X[folds == fold]
        y_valid = y[folds == fold]

        scaler_path = RapidsSvcBaselineCFG.output_dir / f"scaler_seed_{seed}_fold{fold}.pkl"
        scaler = pickle.load(open(scaler_path, "rb"))

        model_path = RapidsSvcBaselineCFG.output_dir / f"model_seed_{seed}_fold{fold}.pkl"
        model = pickle.load(open(model_path, "rb"))

        X_valid = scaler.transform(X_valid)
        y_valid_pred = model.predict(X_valid)

        train_df = train_df.join(
            train_df.filter(
                pol.col("fold") == fold
            ).select(
                pol.col("image_id"),
            ).with_column(
                pol.lit(y_valid_pred.reshape(-1)).alias("prediction_i")
            ),
            on="image_id",
            how="left"
        ).select([
            pol.col("prediction_id"),
            pol.col("image_id"),
            pol.col("fold"),
            (pol.col("prediction") + pol.col("prediction_i").fill_null(pol.lit(0))).alias("prediction"),
            pol.col("cancer")
        ])

    train_df.write_csv(RapidsSvcBaselineCFG.output_dir / "oof_before_agg.csv")
    oof_df = oof_df.filter(
        pol.col("fold").is_in(GeneralCFG.train_fold)
    ).select([
        pol.col("prediction_id"),
        pol.col("fold"),
    ]).join(
        train_df.groupby("prediction_id").agg([
            pol.col("prediction").mean().alias("prediction"),
            pol.col("cancer").mean().alias("cancer")
        ]),
        on="prediction_id",
        how="left"
    )

    oof_df.write_csv(RapidsSvcBaselineCFG.output_dir / "oof.csv")
    whole_metrics, metrics_by_folds, metrics_each_fold = calc_oof_score_pol.calc(oof_df, is_debug=debug, seed=seed, is_sigmoid=True)
    return whole_metrics, metrics_by_folds, metrics_each_fold
