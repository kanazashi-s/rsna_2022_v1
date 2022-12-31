import pandas as pd
from sklearn.model_selection import StratifiedKFold  # TODO: Change to fit the competition
from config.general import GeneralCFG


def add_fold_column(input_df, num_folds: int, random_state: int):
    output_df = input_df.copy()
    output_df['fold'] = -1

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=output_df, y=y)):
        output_df.loc[valid_idx, 'fold'] = fold

    return output_df
