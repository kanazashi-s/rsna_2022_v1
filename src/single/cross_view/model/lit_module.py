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

        self.stage0_backbone = nn.Sequential(
            base_model.conv_stem,
            base_model.bn1,
            base_model.blocks[:4]
        )
        self.stage1_backbone = base_model.blocks[4]
        self.stage2_backbone = base_model.blocks[5]
        
        # self.cvam_stage0 = CrossViewAttentionModule(
        #     in_channels=base_model.blocks[:4][-1][-1].conv_pwl.out_channels,
        # )
        # self.cvam_stage1 = CrossViewAttentionModule(
        #     in_channels=base_model.blocks[4][-1].conv_pwl.out_channels,
        # )
        self.cvam_stage2 = CrossViewAttentionModule(
            in_channels=base_model.blocks[5][-1].conv_pwl.out_channels,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(base_model.blocks[5][-1].conv_pwl.out_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_weights(self.mlp)
        self.loss = self._get_loss()
        self.learning_rate = CrossViewCFG.lr

    def forward(self, inputs):
        x_lcc, x_rcc, x_lmlo, x_rmlo = [image[0] for image in inputs]
        num_lcc = x_lcc.shape[0]
        num_rcc = x_rcc.shape[0]
        num_lmlo = x_lmlo.shape[0]
        num_rmlo = x_rmlo.shape[0]
        x = self._concat_views(x_lcc, x_rcc, x_lmlo, x_rmlo)

        # forward
        x = self.stage0_backbone(x)
        x = self.stage1_backbone(x)
        x = self.stage2_backbone(x)

        # cvam stage2
        x_lcc, x_rcc, x_lmlo, x_rmlo = self._split_views(x, num_lcc, num_rcc, num_lmlo, num_rmlo)
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
        l_output = self.mlp(l_features)
        r_output = self.mlp(r_features)
        outputs = torch.cat([l_output, r_output], dim=1)

        return outputs

    @staticmethod
    def _split_views(x, num_lcc, num_rcc, num_lmlo, num_rmlo):
        x_lcc = x[:num_lcc]
        x_rcc = x[num_lcc:num_lcc + num_rcc]
        x_lmlo = x[num_lcc + num_rcc:num_lcc + num_rcc + num_lmlo]
        x_rmlo = x[num_lcc + num_rcc + num_lmlo:]
        assert x_rmlo.shape[0] == num_rmlo
        return x_lcc, x_rcc, x_lmlo, x_rmlo

    @staticmethod
    def _concat_views(x_lcc, x_rcc, x_lmlo, x_rmlo):
        x = torch.cat([x_lcc, x_rcc, x_lmlo, x_rmlo], dim=0)
        return x

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
    import torchinfo
    data_module = DataModule(seed=42, fold=0, num_workers=0)
    data_module.setup()
    images, labels = next(iter(data_module.train_dataloader()))
    model = LitModel()
    outputs = model(images)
    print(outputs.shape)

    torchinfo.summary(model, input_size=(4, 1, 1, 1, 1536, 1410))
    print(model)