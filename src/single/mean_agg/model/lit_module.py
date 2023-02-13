import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG
from single.mean_agg.model.optimizers import OptimizersMixin
from single.mean_agg.model.train_net import TrainNetMixin
from single.mean_agg.model.vaild_net import ValidNetMixin
from single.mean_agg.model.predict_net import PredictNetMixin
from single.mean_agg.model.test_net import TestNetMixin
# from single.mean_agg.next_vit import next_vit
from utils.model_utils.pl_mixin import InitWeightsMixin
from utils.model_utils.focal_loss import SigmoidFocalLoss
from metrics import get_scores


class LitModel(
    OptimizersMixin,
    TrainNetMixin,
    ValidNetMixin,
    PredictNetMixin,
    TestNetMixin,
    InitWeightsMixin,
):
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
        self._init_weights(self.fc)
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


if __name__ == '__main__':
    model = LitModel()
    print(model)
