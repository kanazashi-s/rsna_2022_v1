import sys
from cfg.general import GeneralCFG
if GeneralCFG.is_kaggle:
    sys.path.append('/kaggle/input/timm-0-6-9/pytorch-image-models-master')

from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_tensorrt
import timm
from single.mean_agg.config import MeanAggCFG
from utils.model_utils.focal_loss import SigmoidFocalLoss
from metrics import get_scores


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # define pretrained model
        self.is_next_vit = MeanAggCFG.model_name.startswith("nextvit")
        if self.is_next_vit:
            model_config = {
                "model_name": MeanAggCFG.model_name,
                "num_classes": 1000,
                "drop": MeanAggCFG.drop_rate,
                # "path_dropout": MeanAggCFG.drop_path_rate,
                "pretrained": False,
            }
            self.backbone = timm.create_model(**model_config)
            large_checkpoint = torch.load("pretrained_weights/nextvit_base_in1k6m_384.pth")
            self.backbone.load_state_dict(large_checkpoint['model'])
            self.backbone.stem[0].conv.in_channels = 1
            self.backbone.stem[0].conv.weight.data = self.backbone.stem[0].conv.weight.data[:, 0:1, :, :]
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            model_config = {
                "model_name": MeanAggCFG.model_name,
                "num_classes": 0,  # to use feature extractor,
                "in_chans": 1,
                "drop_rate": MeanAggCFG.drop_rate,
                "drop_path_rate": MeanAggCFG.drop_path_rate,
            }
            if GeneralCFG.is_kaggle:
                self.backbone = timm.create_model(**model_config, pretrained=False)
            else:
                self.backbone = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(1024 if self.is_next_vit else self.backbone.num_features, 1)
        # self._init_weights(self.fc)
        self.loss = self._get_loss()
        self.learning_rate = MeanAggCFG.lr

    def forward(self, inputs):
        features = self.backbone(inputs)
        output = self.fc(features)
        return output

    @staticmethod
    def _get_loss():
        if MeanAggCFG.loss_function == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(MeanAggCFG.pos_weight))
        elif MeanAggCFG.loss_function == "SigmoidFocalLoss":
            return SigmoidFocalLoss(
                gamma=MeanAggCFG.focal_loss_gamma,
                alpha=MeanAggCFG.focal_loss_alpha,
                sigmoid=True
            )


def litmodule_to_trt(model_dir: Path, seed: int):
    model_dir = model_dir / f"seed{seed}"
    for fold in GeneralCFG.train_fold:
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
        enabled_precisions = {torch.float, torch.half}  # Run with fp16

        trt_ts_module = torch_tensorrt.compile(
            model_torch, inputs=inputs, enabled_precisions=enabled_precisions
        )

        save_dir = Path.cwd() if GeneralCFG.is_kaggle else model_dir
        torch.jit.save(trt_ts_module, save_dir / f"trt_fold{fold}.ts")


if __name__ == "__main__":
    if GeneralCFG.is_kaggle:
        model_dir = Path("/kaggle/input/mean-agg-swa-20230213")
        litmodule_to_trt(model_dir, seed=42)
    else:
        model_dir = Path("/workspace", "output", "single", "mean_agg", "1536_ker_swa")
        litmodule_to_trt(model_dir, seed=42)
