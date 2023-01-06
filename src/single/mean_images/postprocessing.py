import pandas as pd
import numpy as np


def agg_by_prediction_id(valid_df, all_preds, agg_func='max'):
    """
    モデルの予測値を、 prediction_id ごとに集約する

    valid_df: pd.DataFrame
        prediction_id と、 cancer 列を持つデータフレーム
        事前に、 validation_step_outputs から生成した all_labels と、 valid_df の cancer 列が一致することを確認する
    all_preds: np.ndarray
        validation_step_outputs から生成した all_preds
    agg_func: str
        集約関数 (max, mean, median, min)
        デフォルトのままでいいでしょう〜。
    """

    valid_df['predictions'] = all_preds.flatten()
    max_predictions_df = valid_df.groupby('prediction_id').agg({
        'predictions': agg_func,
        'cancer': agg_func
    }).reset_index()

    max_predictions_df = valid_df[["prediction_id"]].drop_duplicates().reset_index(drop=True).merge(
        max_predictions_df,
        on="prediction_id",
        how="left"
    )

    max_predictions = max_predictions_df['predictions'].values
    max_labels = max_predictions_df['cancer'].values

    return max_predictions, max_labels