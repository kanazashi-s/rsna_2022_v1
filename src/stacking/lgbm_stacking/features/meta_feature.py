import polars as pol
from cfg.general import GeneralCFG
from data import load_processed_data_pol


def meta_feature(X_base: pol.DataFrame, seed: int):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    whole_df = load_processed_data_pol.train(seed=seed)
    X_out = X_base.join(
        whole_df.select([
            pol.col("prediction_id"),
            pol.col("fold"),
            pol.col(GeneralCFG.target_col)
        ]).groupby("prediction_id").agg([
            pol.col("fold").max().alias("fold"),
            pol.col(GeneralCFG.target_col).mean().alias(GeneralCFG.target_col),
        ]),
        on="prediction_id",
        how="left"
    ).select([
        pol.col("prediction_id"),
        pol.col("fold"),
        pol.col(GeneralCFG.target_col),
    ])
    return X_out


def age_value(X_base: pol.DataFrame, seed: int):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    whole_df = load_processed_data_pol.train(seed=seed)
    X_out = X_base.join(
        whole_df.select([
            pol.col("prediction_id"),
            pol.col("age"),
        ]).groupby("prediction_id").agg([
            pol.col("age").max().alias("age_value"),
        ]),
        on="prediction_id",
        how="left"
    ).select([
        pol.col("prediction_id"),
        pol.col("age_value"),
    ])
    return X_out
