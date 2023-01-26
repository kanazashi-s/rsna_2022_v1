import numpy as np
from sklearn.metrics import roc_auc_score
from metrics.pfbeta import pfbeta, get_best_pfbeta_thresh, get_f1_curve
from metrics.roc import get_roc_curve, get_auc_roc
from metrics.pr import get_pr_curve, get_auc_pr
from metrics.cm import get_confusion_matrix


def get(y_trues, y_preds, is_sigmoid=False):
    """
    与えられた正解ラベルと予測値から、以下すべての情報を取得し辞書に格納して返す
    * F1 curve 描画・保存して、 matplotlib の figure オブジェクトを格納
    * ROC curve 描画・保存して、 matplotlib の figure オブジェクトを格納
    * PR curve 描画・保存して、 matplotlib の figure オブジェクトを格納
    * Lift chart 描画・保存して、 matplotlib の figure オブジェクトを格納
    * Best pf1 score + threshold
    * auc-roc score
    * auc-pr score
    * Confusion matrix (pfbeta の threshold で2値化した予測値に対して算出)

    Parameters:
    ----------
    y_trues: np.ndarray
        正解ラベル
        Shape: (num_data, )
    y_preds: np.ndarray
        予測値 シグモイド関数実行前の値 (is_sigmoid=False) または シグモイド関数実行後の値 (is_sigmoid=True)
        Shape: (num_data, )
    is_sigmoid: bool
        予測値がシグモイド関数実行後の値かどうか
    """

    assert y_trues.shape == y_preds.shape
    assert np.isin(y_trues, [0, 1]).all()

    if not is_sigmoid:
        y_preds = 1 / (1 + np.exp(-y_preds))

    ret_dict = {}
    ret_dict["f1_curve"] = get_f1_curve(y_trues, y_preds)
    ret_dict["roc_curve"] = get_roc_curve(y_trues, y_preds)
    ret_dict["pr_curve"] = get_pr_curve(y_trues, y_preds)
    # ret_dict["lift_chart"] = get_lift_chart(y_trues, y_preds)  # TODO: 未実装、いつか欲しい
    ret_dict["best_thresh_pfbeta"], ret_dict["best_pfbeta"] = get_best_pfbeta_thresh(y_trues, y_preds)
    ret_dict["auc_roc"] = get_auc_roc(y_trues, y_preds)
    ret_dict["auc_pr"] = get_auc_pr(y_trues, y_preds)

    best_thresh_pfbeta = ret_dict["best_thresh_pfbeta"]
    ret_dict["TP"], ret_dict["FP"], ret_dict["TN"], ret_dict["FN"] = get_confusion_matrix(
        y_trues, y_preds, best_thresh_pfbeta
    )

    return ret_dict
