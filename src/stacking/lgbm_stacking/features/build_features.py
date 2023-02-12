from typing import List, Callable
import polars as pol


def make(use_features: List[Callable], X_base: pol.DataFrame, seed: int):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    X = X_base.clone()
    for use_feature in use_features:
        X_out = use_feature(X_base=X_base, seed=seed)
        X = X.join(X_out, on='prediction_id', how='left')
        # assert (X['combining_shaking_id'] == X_base).sum() == len(X_base)
        assert X.get_column("prediction_id").series_equal(X_base.get_column("prediction_id"))
        assert len(X) == len(X_base)

    return X
