import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from cfg.general import GeneralCFG
from single.two_view_concat.config import TwoViewConcatCFG
from single.two_view_concat.model.optimizers import OptimizersMixin
from single.two_view_concat.model.train_net import TrainNetMixin
from single.two_view_concat.model.vaild_net import ValidNetMixin
from single.two_view_concat.model.predict_net import PredictNetMixin
from single.two_view_concat.model.test_net import TestNetMixin
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
        model_config = {
            "model_name": TwoViewConcatCFG.model_name,
            "num_classes": 0,  # to use feature extractor,
            "in_chans": 1,
            "drop_rate": TwoViewConcatCFG.drop_rate,
            "drop_path_rate": TwoViewConcatCFG.drop_path_rate,
        }
        if GeneralCFG.is_kaggle:
            self.backbone = timm.create_model(**model_config, pretrained=False)
        else:
            self.backbone = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(self.backbone.num_features * 2, 1)
        self._init_weights(self.fc)
        self.loss = self._get_loss()
        self.learning_rate = TwoViewConcatCFG.lr

    def forward(self, inputs):
        mlo_image, cc_image = inputs
        mlo_features = self.backbone(mlo_image)
        cc_features = self.backbone(cc_image)
        features = torch.cat([mlo_features, cc_features], dim=1)
        output = self.fc(features)
        return output

    @staticmethod
    def _get_loss():
        if TwoViewConcatCFG.loss_function == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(TwoViewConcatCFG.pos_weight))
        elif TwoViewConcatCFG.loss_function == "SigmoidFocalLoss":
            return SigmoidFocalLoss(
                gamma=TwoViewConcatCFG.focal_loss_gamma,
                alpha=TwoViewConcatCFG.focal_loss_alpha,
                sigmoid=True
            )


if __name__ == '__main__':
    model = LitModel()
    print(model)
