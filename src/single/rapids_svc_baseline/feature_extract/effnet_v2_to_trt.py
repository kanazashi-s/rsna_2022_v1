import sys
import os
is_kaggle = bool(int(os.environ["IS_KAGGLE_ENVIRONMENT"]))
if is_kaggle:
    sys.path.append('/kaggle/input/timm-0-6-9/pytorch-image-models-master')

from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_tensorrt
import timm


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_m_in21ft1k",
            pretrained=True,
            num_classes=0,
            in_chans=1,
        )

    def forward(self, inputs):
        features = self.backbone(inputs)
        return features


def litmodule_to_trt(model_dir, seed: int = 42):
    model_dir = model_dir / f"seed{seed}"
    model_torch = MyModel()
    model_torch.eval().cuda()

    inputs = [
        torch_tensorrt.Input(
            shape=[32, 1, 1536, 1410],
            dtype=torch.float,
        )
    ]
    enabled_precisions = {torch.float, torch.half}  # Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        model_torch, inputs=inputs, enabled_precisions=enabled_precisions
    )

    save_dir = Path.cwd() / f"seed{seed}" if is_kaggle else model_dir
    save_dir.mkdir(exist_ok=True, parents=True)
    torch.jit.save(trt_ts_module, save_dir / f"trt_effnet_v2_m.ts")


if __name__ == "__main__":
    if is_kaggle:
        model_dir = Path("/kaggle/input/mean-agg-ker-optim-swa-20220213")
        litmodule_to_trt(model_dir, seed=42)
    else:
        from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
        model_dir = RapidsSvcBaselineCFG.output_dir
        litmodule_to_trt(model_dir, seed=42)
