import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

from pathlib import Path
from tqdm import tqdm
import polars as pol
import torch
import torch.nn as nn
from torch.cuda import amp
import torch_tensorrt
import timm
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG
from single.rapids_svc_baseline.feature_extract.dataset import get_extract_dataloader


def extract(base_df, use_saved_features=True):
    """Extract features from EfficientNetV2-M model.
    Returns: pol.DataFrame
    """

    if use_saved_features:
        feature_df = pol.read_csv(RapidsSvcBaselineCFG.output_dir / "feature_extract" / "pretrain_effnet_v2_s.csv")
        return feature_df

    out_feature_df = base_df.clone().with_column(
        pol.lit(0).alias("features")
    )

    for fold in GeneralCFG.train_fold:
        input_dir = MeanAggCFG.trt_model_dir / f"seed42"
        trt_ts_module = torch.jit.load(input_dir / f"trt_fold{fold}.ts")

        fold_base_df = base_df.filter(pol.col("fold") == fold)
        dataloader = get_extract_dataloader(fold_base_df, batch_size=8)

        # torch のモデルを使って、 test_df の prediction を予測
        features = []
        for batch in dataloader:
            batch = batch.cuda().half()
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    features.append(trt_ts_module(batch))

        features = torch.cat(features)
        features = features.detach().cpu().numpy()

        out_feature_df = out_feature_df.join(
            out_feature_df.filter(
                pol.col("fold") == fold
            ).select(
                pol.col("image_id")
            ).with_columns(
                pol.lit(features).alias("features_i")
            ),
            on="image_id",
            how="left"
        ).select([
            pol.col("image_id"),
            pol.col("fold"),
            (pol.col("features") + pol.col("features_i").fill_null(pol.lit(0))).alias("features"),
        ])

    return out_feature_df


if __name__ == "__main__":
    from data import load_processed_data_pol
    from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
    MeanAggCFG.trt_model_dir = MeanAggCFG.trt_model_dir / "1536_ker_swa"
    train_df = load_processed_data_pol.train(42)[:100]
    feature_df = extract(train_df, use_saved_features=False)
    (RapidsSvcBaselineCFG.output_dir / "feature_extract").mkdir(exist_ok=True, parents=True)
    feature_df.write_csv(RapidsSvcBaselineCFG.output_dir / "feature_extract" / "pretrain_effnet_v2_s.csv")
