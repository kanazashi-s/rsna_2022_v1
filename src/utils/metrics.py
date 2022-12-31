import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from config.general import GeneralCFG
from data import load_processed_data


def mcrmse(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score


def get_score(y_trues, y_preds):
    score = mcrmse(y_trues, y_preds)
    return score


def get_oof_score(oof_df, seed, is_debug=False):
    train_df = load_processed_data.train(seed=seed)
    if is_debug:
        train_df = pd.concat([
            train_df.loc[train_df["fold"] == fold].head(GeneralCFG.num_use_data) for fold in GeneralCFG.train_fold
        ]).reset_index(drop=True)
    labels = train_df[GeneralCFG.target_col].values
    preds = oof_df[GeneralCFG.target_col].values
    score = get_score(labels, preds)
    return score


if __name__ == "__main__":
    oof_df = load_processed_data.sample_oof(seed=seed)
    score = get_oof_score(oof_df, 42)
    print(score)
