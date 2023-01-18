import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(y_trues, y_preds, threshold=None) -> (int, int, int, int):
    """
    与えられた正解ラベルと予測値から、2値化した予測値を用いて混同行列を算出する

    Parameters:
    ----------
    y_trues: np.ndarray
        正解ラベル
        Shape: (num_data, )
    y_preds: np.ndarray
        予測値 シグモイド関数実行後の値
        Shape: (num_data, )
    threshold: float
        2値化する際の閾値
    """
    if threshold is not None:
        y_preds = (y_preds >= threshold).astype(np.int)

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    return tp, fp, tn, fn
