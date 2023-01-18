import numpy as np
import pandas as pd
from cfg.general import GeneralCFG
from data import load_processed_data
from metrics import get_scores
from metrics.pfbeta import pfbeta
from metrics.roc import get_auc_roc
from metrics.pr import get_auc_pr
from metrics.cm import get_confusion_matrix


def calc(oof_df, seed, is_debug=False):
    """
    OOFデータフレームから、以下すべての情報を取得し辞書に格納して返す
    * 全体の F1 Curve
    * 全体の ROC Curve
    * 全体の PR Curve
    * 全体の Best F1 Score (F1 Score, Threshold)
    * 全体の ROC AUC Score
    * 全体の PR AUC Score
    """

    train_df = load_processed_data.train(seed=seed)
    train_df = train_df.loc[train_df["view"].isin(["MLO", "CC"])].reset_index(drop=True)
    train_df = train_df.groupby("prediction_id")[[GeneralCFG.target_col, "fold"]].max().reset_index()

    if is_debug:
        train_df = oof_df[["prediction_id"]].merge(
            train_df, on="prediction_id", how="left"
        ).reset_index(drop=True)

    else:
        train_df = load_processed_data.sample_oof(seed=seed)[["prediction_id"]].merge(
            train_df, on="prediction_id", how="left"
        ).reset_index(drop=True)

    assert oof_df["prediction_id"].equals(train_df["prediction_id"])
    assert oof_df["fold"].equals(train_df["fold"])

    # if is_debug:
    #     train_df = pd.concat([
    #         train_df.loc[train_df["fold"] == fold].head(GeneralCFG.num_use_data) for fold in GeneralCFG.train_fold
    #     ]).reset_index(drop=True)

    labels = train_df[GeneralCFG.target_col].values
    preds = oof_df[GeneralCFG.target_col].values
    whole_metrics = get_scores.get(labels, preds)

    metrics_each_fold = []
    fold_pred_oof_df = oof_df.copy()  # 各Foldのthreshで2値化した予測値を格納する

    for fold in GeneralCFG.train_fold:
        fold_labels = train_df.loc[train_df["fold"] == fold][GeneralCFG.target_col].values
        fold_preds = oof_df.loc[oof_df["fold"] == fold][GeneralCFG.target_col].values
        fold_score_dict = get_scores.get(fold_labels, fold_preds)
        metrics_each_fold.append(fold_score_dict)
        fold_thresh = fold_score_dict["best_thresh_pfbeta"]
        fold_preds_sigmoid = 1 / (1 + np.exp(-fold_preds))
        fold_pred_oof_df.loc[fold_pred_oof_df["fold"] == fold, GeneralCFG.target_col] = (fold_preds_sigmoid >= fold_thresh).astype(int)
        print(f"fold {fold}: score={fold_score_dict['best_pfbeta']:.4f}, auc_roc={fold_score_dict['auc_roc']:.4f}, thresh={fold_score_dict['best_thresh_pfbeta']:.4f}")

    # 各Foldのthreshで2値化した予測値
    preds_by_folds = fold_pred_oof_df[GeneralCFG.target_col].values
    # metrics_by_folds = get_scores.get(labels, 1 / (1 + np.exp(-preds_by_folds)))
    metrics_by_folds = {
        "pfbeta": pfbeta(labels, preds_by_folds),
        "auc_roc": get_auc_roc(labels, preds_by_folds),
        "auc_pr": get_auc_pr(labels, preds_by_folds),
    }
    cm_by_folds = get_confusion_matrix(labels, preds_by_folds)
    metrics_by_folds["TP"], metrics_by_folds["FP"], metrics_by_folds["TN"], metrics_by_folds["FN"] = cm_by_folds

    return whole_metrics, metrics_by_folds, metrics_each_fold
