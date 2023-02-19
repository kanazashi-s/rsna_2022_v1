import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from cfg.general import GeneralCFG
from single.cross_view.config import CrossViewCFG
from single.cross_view.model.optimizers import OptimizersMixin
from single.cross_view.model.train_net import TrainNetMixin
from single.cross_view.model.vaild_net import ValidNetMixin
from single.cross_view.model.predict_net import PredictNetMixin
from single.cross_view.model.test_net import TestNetMixin
from utils.model_utils.pl_mixin import InitWeightsMixin
from utils.model_utils.focal_loss import SigmoidFocalLoss


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

        model_config = {
            "model_name": CrossViewCFG.model_name,
            "num_classes": 0,  # to use feature extractor,
            "in_chans": 1,
            "drop_rate": CrossViewCFG.drop_rate,
            "drop_path_rate": CrossViewCFG.drop_path_rate,
        }
        if GeneralCFG.is_kaggle:
            self.backbone = timm.create_model(**model_config, pretrained=False)
        else:
            self.backbone = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(self.backbone.num_features, 1)
        self._init_weights(self.fc)
        self.loss = self._get_loss()
        self.learning_rate = CrossViewCFG.lr

    def forward(self, inputs):
        features = self.backbone(inputs)
        output = self.fc(features)
        return output

    @staticmethod
    def _get_loss():
        if CrossViewCFG.loss_function == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CrossViewCFG.pos_weight))
        elif CrossViewCFG.loss_function == "SigmoidFocalLoss":
            return SigmoidFocalLoss(
                gamma=CrossViewCFG.focal_loss_gamma,
                alpha=CrossViewCFG.focal_loss_alpha,
                sigmoid=True
            )


if __name__ == '__main__':
    model = LitModel()
    print(model)
