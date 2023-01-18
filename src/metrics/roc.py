import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import RocCurveDisplay


def get_roc_curve(y_trues: np.ndarray, y_preds: np.ndarray) -> plt.Figure:
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)

    Returns:
        plt.Figure
    """
    assert y_trues.shape == y_preds.shape
    assert np.isin(y_trues, [0, 1]).all()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    fig = display.plot().figure_
    return fig


def get_auc_roc(y_trues: np.ndarray, y_preds: np.ndarray) -> float:
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)
    """
    assert y_trues.shape == y_preds.shape
    assert np.isin(y_trues, [0, 1]).all()

    if y_trues.sum() == 0:
        return 0.0

    auc_roc = roc_auc_score(y_trues, y_preds)
    return auc_roc


if __name__ == "__main__":
    y_trues = np.array([0, 0, 1, 1])
    y_preds = np.array([0.1, 0.4, 0.35, 0.8])
    auc_roc = get_auc_roc(y_trues, y_preds)
    fig = get_roc_curve(y_trues, y_preds)
    print(auc_roc)
