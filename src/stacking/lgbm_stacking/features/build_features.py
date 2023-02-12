from typing import List, Callable
import polars as pol


def make(use_features: List[Callable], X_base: pol.DataFrame):
    assert isinstance(X_base, pol.DataFrame)
    assert X_base.columns == ["prediction_id"]

    X = X_base.copy()
    for use_feature in use_features:
        X_out = use_feature(X_base)
        X = X.join(X_out, on='prediction_id', how='left')
        # assert (X['combining_shaking_id'] == X_base).sum() == len(X_base)
        assert X.columns == X_base.columns + X_out.columns
        assert X.get_columns("prediction_id").series_equal(X_base.get_columns("prediction_id"))
        assert len(X) == len(X_base)

    return X
