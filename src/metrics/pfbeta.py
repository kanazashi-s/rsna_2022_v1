import numpy as np
import matplotlib.pyplot as plt


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


def get_best_pfbeta_thresh(y_trues, y_preds):
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)
    """

    threshs = np.linspace(0, 1, 101)
    f1s = [pfbeta(y_trues, (y_preds > thr).astype(float)) for thr in threshs]
    idx = np.argmax(f1s)
    thresh, best_pfbeta = threshs[idx], f1s[idx]
    return thresh, best_pfbeta


def get_f1_curve(y_trues, y_preds) -> plt.Figure:
    """
    y_trues: np.array (0 or 1)
    y_preds: np.array (0~1)
    """
    threshs = np.linspace(0, 1, 101)
    f1s = [pfbeta(y_trues, (y_preds > thr).astype(float)) for thr in threshs]
    fig = plt.figure()
    plt.plot(threshs, f1s)
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 score curve")
    return fig
