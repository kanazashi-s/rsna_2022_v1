import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from cfg.general import GeneralCFG
from data import load_processed_data


def pfbeta(labels, preds, beta=1):
    assert labels.shape == preds.shape
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


def get_score(y_trues, y_preds):
    """
    与えられた正解ラベルと予測値から、以下の全てのスコアを計算して返す
    * PFBeta値
    * AUC
    * PFBeta計算時の最適な閾値

    Parameters:
    ----------
    y_trues: np.ndarray
        正解ラベル
        Shape: (num_data, )
    y_preds: np.ndarray
        予測値 シグモイド関数実行前の値
        Shape: (num_data, )
    """
    assert y_trues.shape == y_preds.shape
    # y_trues が 0 と 1 以外の値を持たないか確認
    assert np.isin(y_trues, [0, 1]).all()

    y_preds = 1 / (1 + np.exp(-y_preds))
    thresh, best_pfbeta = get_best_thresh(y_trues, y_preds)
    y_preds = (y_preds >= thresh).astype(int)
    score = pfbeta(y_trues, y_preds)

    if y_trues.sum() == 0:
        auc = 0.0
    else:
        auc = roc_auc_score(y_trues, y_preds)

    return score, auc, thresh


def get_oof_score(oof_df, seed, is_debug=False):
    """
    OOFデータフレームから、以下のスコアを計算して返す
    * 全体のPFBeta値
    * 全体のAUC
    * PFBeta計算時の最適な閾値
    * 各クラスのPFBeta値 (最適な閾値を用いて計算)
    * 各クラスのAUC
    """

    train_df = load_processed_data.train(seed=seed)
    train_df = train_df.loc[train_df["view"].isin(["MLO", "CC"])].reset_index(drop=True)
    train_df = train_df.groupby("prediction_id")[[GeneralCFG.target_col, "fold"]].max().reset_index()

    if is_debug:
        train_df = pd.concat([
            train_df.loc[train_df["fold"] == fold].head(GeneralCFG.num_use_data) for fold in GeneralCFG.train_fold
        ]).reset_index(drop=True)

    labels = train_df[GeneralCFG.target_col].values
    preds = oof_df[GeneralCFG.target_col].values
    score, auc, thresh = get_score(labels, preds)

    fold_scores = []
    fold_aucs = []
    fold_threshs = []
    fold_pred_oof_df = oof_df.copy()  # 各Foldのthreshで2値化した予測値を格納する
    for fold in GeneralCFG.train_fold:
        fold_labels = train_df.loc[train_df["fold"] == fold][GeneralCFG.target_col].values
        fold_preds = oof_df.loc[oof_df["fold"] == fold][GeneralCFG.target_col].values
        fold_score, fold_auc, fold_thresh = get_score(fold_labels, fold_preds)
        fold_pred_oof_df.loc[fold_pred_oof_df["fold"] == fold, GeneralCFG.target_col] = (fold_preds >= fold_thresh).astype(int)
        fold_scores.append(fold_score)
        fold_aucs.append(fold_auc)
        fold_threshs.append(fold_thresh)
        print(f"fold {fold}: score={fold_score:.4f}, auc={fold_auc:.4f}, thresh={fold_thresh:.4f}")

    # 各Foldのthreshで2値化した予測値
    preds_2 = fold_pred_oof_df[GeneralCFG.target_col].values
    score_2, auc_2, thresh_2 = get_score(labels, 1 / (1 + np.exp(-preds_2)))

    return score, auc, thresh, fold_scores, fold_aucs, fold_threshs, score_2, auc_2, thresh_2


def get_best_thresh(y_trues, y_preds):
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)
    """

    threshs = np.linspace(0, 1, 101)
    f1s = [pfbeta(y_trues, (y_preds > thr).astype(float)) for thr in threshs]
    idx = np.argmax(f1s)
    thresh, best_pfbeta = threshs[idx], f1s[idx]
    return thresh, best_pfbeta


if __name__ == "__main__":
    oof_df = load_processed_data.sample_oof(seed=42)
    score, auc, thresh, fold_scores, fold_aucs = get_oof_score(oof_df, 42)
    print(score)
