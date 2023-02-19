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

        # define pretrained model
        model_config = {
            "model_name": "efficientnetv2_rw_s",
            "num_classes": 0,  # to use feature extractor,
            "in_chans": 1,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
        }
        if is_kaggle:
            self.backbone = timm.create_model(**model_config, pretrained=False)
        else:
            self.backbone = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, inputs):
        features = self.backbone(inputs)
        output = self.fc(features)
        return output


def litmodule_to_trt(model_dir: Path, seed: int):
    model_dir = model_dir / f"seed{seed}"
    for fold in range(5):
        # lightning のモデルを、 checkpoint から読み込み、 torch のモデルに変換
        state_dict = torch.load(model_dir / f"best_loss_fold{fold}.ckpt")["state_dict"]
        model_torch = MyModel()
        model_torch.load_state_dict(state_dict)

        model_torch.eval()

        inputs = [
            torch_tensorrt.Input(
                min_shape=[1, 1, 1536, 1410],
                opt_shape=[8, 1, 1536, 1410],
                max_shape=[8, 1, 1536, 1410],
                dtype=torch.half,
            )
        ]
        enabled_precisions = {torch.half}  # Run with fp16

        trt_ts_module = torch_tensorrt.compile(
            model_torch, inputs=inputs, enabled_precisions=enabled_precisions
        )

        save_dir = Path.cwd() / f"seed{seed}" if is_kaggle else model_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.jit.save(trt_ts_module, save_dir / f"trt_fold{fold}.ts")


if __name__ == "__main__":
    if is_kaggle:
        model_dir = Path("/kaggle/input/cross-view-ker-optim-swa-20220213")
        litmodule_to_trt(model_dir, seed=42)
    else:
        model_dir = Path("/workspace", "output", "single", "cross_view", "1536_ker_swa")
        litmodule_to_trt(model_dir, seed=42)
