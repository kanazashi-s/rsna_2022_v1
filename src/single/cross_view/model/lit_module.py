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
from single.cross_view.model.cross_view_attention_module import CrossViewAttentionModule
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
            base_model = timm.create_model(**model_config, pretrained=False)
        else:
            base_model = timm.create_model(**model_config, pretrained=True)

        self.stage0_lcc = nn.Sequential(
            base_model.conv_stem,
            base_model.bn1,
            base_model.blocks[:4]
        )
        self.stage1_lcc = base_model.blocks[4]
        self.stage2_lcc = base_model.blocks[5]

        self.stage0_rcc = nn.Sequential(
            base_model.conv_stem,
            base_model.bn1,
            base_model.blocks[:4]
        )
        self.stage1_rcc = base_model.blocks[4]
        self.stage2_rcc = base_model.blocks[5]

        self.stage0_lmlo = nn.Sequential(
            base_model.conv_stem,
            base_model.bn1,
            base_model.blocks[:4]
        )
        self.stage1_lmlo = base_model.blocks[4]
        self.stage2_lmlo = base_model.blocks[5]

        self.stage0_rmlo = nn.Sequential(
            base_model.conv_stem,
            base_model.bn1,
            base_model.blocks[:4]
        )
        self.stage1_rmlo = base_model.blocks[4]
        self.stage2_rmlo = base_model.blocks[5]
        
        self.cvam_stage0 = CrossViewAttentionModule(
            in_channels=base_model.blocks[:4][-1][-1].conv_pwl.out_channels,
        )
        self.cvam_stage1 = CrossViewAttentionModule(
            in_channels=base_model.blocks[4][-1].conv_pwl.out_channels,
        )
        self.cvam_stage2 = CrossViewAttentionModule(
            in_channels=base_model.blocks[5][-1].conv_pwl.out_channels,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.l_mlp = nn.Sequential(
            nn.Linear(base_model.blocks[5][-1].conv_pwl.out_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.r_mlp = nn.Sequential(
            nn.Linear(base_model.blocks[5][-1].conv_pwl.out_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_weights(self.l_mlp)
        self._init_weights(self.r_mlp)
        self.loss = self._get_loss()
        self.learning_rate = CrossViewCFG.lr

    def forward(self, inputs):
        x_lcc, x_rcc, x_lmlo, x_rmlo = [image[0] for image in inputs]

        # stage0 forward
        x_lcc = self.stage0_lcc(x_lcc)
        x_rcc = self.stage0_rcc(x_rcc)
        x_lmlo = self.stage0_lmlo(x_lmlo)
        x_rmlo = self.stage0_rmlo(x_rmlo)

        # cvam stage0
        x_lcc, x_rcc, x_lmlo, x_rmlo = self.cvam_stage0(x_lcc, x_rcc, x_lmlo, x_rmlo)

        # stage1 forward
        x_lcc = self.stage1_lcc(x_lcc)
        x_rcc = self.stage1_rcc(x_rcc)
        x_lmlo = self.stage1_lmlo(x_lmlo)
        x_rmlo = self.stage1_rmlo(x_rmlo)

        # cvam stage1
        x_lcc, x_rcc, x_lmlo, x_rmlo = self.cvam_stage1(x_lcc, x_rcc, x_lmlo, x_rmlo)

        # stage2 forward
        x_lcc = self.stage2_lcc(x_lcc)
        x_rcc = self.stage2_rcc(x_rcc)
        x_lmlo = self.stage2_lmlo(x_lmlo)
        x_rmlo = self.stage2_rmlo(x_rmlo)

        # cvam stage2
        x_lcc, x_rcc, x_lmlo, x_rmlo = self.cvam_stage2(x_lcc, x_rcc, x_lmlo, x_rmlo)

        # global pool
        x_lcc = self.global_pool(x_lcc)
        x_rcc = self.global_pool(x_rcc)
        x_lmlo = self.global_pool(x_lmlo)
        x_rmlo = self.global_pool(x_rmlo)

        # concat
        if x_lcc.size(0) != 1:
            x_lcc = x_lcc.mean(dim=0, keepdim=True)
        if x_rcc.size(0) != 1:
            x_rcc = x_rcc.mean(dim=0, keepdim=True)
        if x_lmlo.size(0) != 1:
            x_lmlo = x_lmlo.mean(dim=0, keepdim=True)
        if x_rmlo.size(0) != 1:
            x_rmlo = x_rmlo.mean(dim=0, keepdim=True)

        l_features = torch.cat([x_lcc, x_rcc], dim=1)
        l_features = l_features.view(l_features.size(0), -1)
        r_features = torch.cat([x_lmlo, x_rmlo], dim=1)
        r_features = r_features.view(r_features.size(0), -1)

        # mlp
        l_output = self.l_mlp(l_features)
        r_output = self.r_mlp(r_features)
        outputs = torch.cat([l_output, r_output], dim=1)

        return outputs

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
    from single.cross_view.data_module import DataModule
    data_module = DataModule(seed=42, fold=0, num_workers=0)
    data_module.setup()
    images, labels = next(iter(data_module.train_dataloader()))
    model = LitModel()
    outputs = model(images)
    print(outputs.shape)