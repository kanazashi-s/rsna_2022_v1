from pathlib import Path
import polars as pol
from cfg.general import GeneralCFG
from data import load_processed_data_pol


def model_output_agg(X_base, seed):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    model_output_path = Path("/workspace", "output", "single", "mean_agg", "1536_ker_swa")
    model_output_df = pol.read_csv(model_output_path / "oof_before_agg.csv")
    X_out = X_base.join(
        model_output_df.select([
            pol.col("prediction_id"),
            pol.col("prediction"),
        ]).groupby("prediction_id").agg([
            pol.col("prediction").max().alias("prediction_max"),
            pol.col("prediction").min().alias("prediction_min"),
            pol.col("prediction").mean().alias("prediction_mean"),
            pol.col("prediction").median().alias("prediction_median"),
        ]),
        on="prediction_id",
        how="left"
    ).select([
        pol.col("prediction_id"),
        pol.col("prediction_max"),
        pol.col("prediction_min"),
        pol.col("prediction_mean"),
        pol.col("prediction_median"),
    ])
    return X_out