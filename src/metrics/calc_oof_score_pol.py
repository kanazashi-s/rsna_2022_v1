import numpy as np
import polars as pol
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from metrics import get_scores
from metrics.pfbeta import pfbeta
from metrics.roc import get_auc_roc
from metrics.pr import get_auc_pr
from metrics.cm import get_confusion_matrix


def calc(oof_df: pol.DataFrame, seed: int, is_sigmoid: bool, is_debug=False):
    """
    OOFデータフレームから、以下すべての情報を取得し辞書に格納して返す
    * 全体の F1 Curve
    * 全体の ROC Curve
    * 全体の PR Curve
    * 全体の Best F1 Score (F1 Score, Threshold)
    * 全体の ROC AUC Score
    * 全体の PR AUC Score
    """

    if is_debug:
        train_df = load_processed_data_pol.debug_train(seed=seed)
    else:
        train_df = load_processed_data_pol.train(seed=seed)

    train_df = oof_df.select("prediction_id").join(
        train_df.filter(
            pol.col("fold").is_in(GeneralCFG.train_fold)
        ).select([
            pol.col("prediction_id"),
            pol.col("fold"),
            pol.col(GeneralCFG.target_col),
        ]).groupby("prediction_id").agg([
            pol.col(GeneralCFG.target_col).mean().alias(GeneralCFG.target_col),
            pol.col("fold").mean().alias("fold"),
        ]),
        on="prediction_id",
        how="left"
    )

    assert oof_df.get_column("prediction_id").series_equal(train_df.get_column("prediction_id"))
    assert (train_df.get_column(GeneralCFG.target_col) - train_df.get_column(GeneralCFG.target_col).cast(int) == 0).all()

    labels = train_df.get_column(GeneralCFG.target_col).to_numpy()
    preds = oof_df.get_column("prediction").to_numpy()

    assert preds.min() >= 0 and preds.max() <= 1

    whole_metrics = get_scores.get(labels, preds, is_sigmoid=is_sigmoid)

    metrics_each_fold = []
    fold_pred_oof_df = oof_df.clone()  # 各Foldのthreshで2値化した予測値を格納する
    fold_pred_oof_df = fold_pred_oof_df.with_column(
        pol.lit(0).alias("prediction")
    )

    for fold in GeneralCFG.train_fold:
        fold_labels = train_df.filter(train_df["fold"] == fold).get_column(GeneralCFG.target_col).to_numpy()
        fold_preds = oof_df.filter(oof_df["fold"] == fold).get_column("prediction").to_numpy()
        fold_score_dict = get_scores.get(fold_labels, fold_preds, is_sigmoid=is_sigmoid)
        metrics_each_fold.append(fold_score_dict)
        fold_thresh = fold_score_dict["best_thresh_pfbeta"]
        if not is_sigmoid:
            fold_preds_sigmoid = 1 / (1 + np.exp(-fold_preds))
        else:
            fold_preds_sigmoid = fold_preds

        fold_pred_oof_df = fold_pred_oof_df.join(
            oof_df.filter(
                pol.col("fold") == fold
            ).select(
                pol.col("prediction_id")
            ).with_column(
                pol.lit(fold_preds_sigmoid >= fold_thresh).cast(pol.Boolean).alias("prediction_i")
            ),
            on="prediction_id",
            how="left",
        ).select([
            pol.col("prediction_id"),
            pol.col("fold"),
            (pol.col("prediction") + pol.col("prediction_i").fill_null(pol.lit(0))).alias("prediction"),
            pol.col(GeneralCFG.target_col),
        ])

        print(f"fold {fold}: score={fold_score_dict['best_pfbeta']:.4f}, auc_roc={fold_score_dict['auc_roc']:.4f}, thresh={fold_score_dict['best_thresh_pfbeta']:.4f}")

    # 各Foldのthreshで2値化した予測値
    preds_by_folds = fold_pred_oof_df.get_column("prediction").to_numpy()
    metrics_by_folds = {
        "pfbeta": pfbeta(labels, preds_by_folds),
        "auc_roc": get_auc_roc(labels, preds_by_folds),
        "auc_pr": get_auc_pr(labels, preds_by_folds),
    }
    cm_by_folds = get_confusion_matrix(labels, preds_by_folds)
    metrics_by_folds["TP"], metrics_by_folds["FP"], metrics_by_folds["TN"], metrics_by_folds["FN"] = cm_by_folds

    return whole_metrics, metrics_by_folds, metrics_each_fold
