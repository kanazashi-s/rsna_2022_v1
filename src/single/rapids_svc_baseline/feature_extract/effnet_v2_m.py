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
from single.rapids_svc_baseline.feature_extract.dataset import get_extract_dataloader


def extract(base_df, use_saved_features=True):
    """Extract features from EfficientNetV2-M model.
    Returns: pol.DataFrame
    """

    if use_saved_features:
        feature_df = pol.read_csv(RapidsSvcBaselineCFG.output_dir / "feature_extract" / "effnet_v2_m.csv")
        return feature_df

    model = timm.create_model(
        "tf_efficientnetv2_m_in21ft1k",
        pretrained=True,
        num_classes=0,
        in_chans=1,
    ).cuda().half()
    dataloader = get_extract_dataloader(base_df, batch_size=8)

    features = []
    for batch in tqdm(dataloader):
        batch = batch.cuda().half()
        with torch.no_grad():
            with amp.autocast(enabled=True):
                feature = model(batch)
                features.append(feature)
    features = torch.cat(features)
    features = features.detach().cpu().numpy()

    feature_df = pol.concat([
        base_df["image_id"],
        pol.DataFrame(features),
    ], how="horizontal")
    return feature_df


if __name__ == "__main__":
    from data import load_processed_data_pol
    from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
    train_df = load_processed_data_pol.train(42)
    feature_df = extract(train_df, use_saved_features=False)
    feature_df.write_csv(RapidsSvcBaselineCFG.output_dir / "feature_extract" / "effnet_v2_m.csv")
