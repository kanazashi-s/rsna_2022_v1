import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from cfg.general import GeneralCFG


def add_fold_column(input_df, num_folds: int, random_state: int):
    output_df = input_df.copy()
    output_df['fold'] = -1

    kf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    splits = kf.split(X=output_df, y=output_df["cancer"], groups=output_df["patient_id"])
    for fold, (train_idx, valid_idx) in enumerate(splits):
        output_df.loc[valid_idx, 'fold'] = fold

    return output_df
