import polars as pol
from cfg.general import GeneralCFG
from data import load_processed_data_pol


def count_machine_id(X_base: pol.DataFrame, seed: int):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    whole_df = load_processed_data_pol.train(seed=seed)
    X_out = X_base.join(
        whole_df.select([
            pol.col("prediction_id"),
            pol.col("machine_id"),
        ]).groupby("prediction_id").agg(
            pol.col("machine_id").max().alias("machine_id")
        ).join(
            whole_df.get_column("machine_id").value_counts(),
            on="machine_id",
            how="left"
        ),
        on="prediction_id",
        how="left"
    ).select([
        pol.col("prediction_id"),
        pol.col("counts").alias("count_machine_id"),
    ])
    return X_out


if __name__ == "__main__":
    X_base = load_processed_data_pol.sample_oof(seed=42).select(["prediction_id"])
    X_out = count_machine_id(X_base, seed=42)
    print(X_out.head(10))