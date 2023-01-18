import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import PrecisionRecallDisplay


def get_pr_curve(y_trues: np.ndarray, y_preds: np.ndarray) -> plt.Figure:
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)

    Returns:
        plt.Figure
    """
    assert y_trues.shape == y_preds.shape
    assert np.isin(y_trues, [0, 1]).all()

    precision, recall, thresholds = precision_recall_curve(y_trues, y_preds)
    pr_auc = auc(recall, precision)
    display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc)
    fig = display.plot().figure_
    return fig


def get_auc_pr(y_trues: np.ndarray, y_preds: np.ndarray) -> float:
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)
    """
    assert y_trues.shape == y_preds.shape
    assert np.isin(y_trues, [0, 1]).all()

    if y_trues.sum() == 0:
        return 0.0

    precision, recall, thresholds = precision_recall_curve(y_trues, y_preds)
    auc_pr = auc(recall, precision)
    return auc_pr


if __name__ == "__main__":
    y_trues = np.array([0, 0, 1, 1])
    y_preds = np.array([0.1, 0.4, 0.35, 0.8])
    auc_pr = get_auc_pr(y_trues, y_preds)
    fig = get_pr_curve(y_trues, y_preds)
    print(auc_pr)