import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from transformers import get_cosine_schedule_with_warmup
from cfg.general import GeneralCFG
from single.last_2_images.config import Last2ImagesCFG
from single.model_utils.focal_loss import SigmoidFocalLoss
from utils.metrics import get_score


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # define pretrained model
        model_config = {
            "model_name": Last2ImagesCFG.model_name,
            "num_classes": 0,  # to use feature extractor,
            "in_chans": 1,
        }
        if GeneralCFG.is_kaggle:
            self.backbone_mlo = timm.create_model(**model_config, pretrained=False)
            self.backbone_cc = timm.create_model(**model_config, pretrained=False)
        else:
            self.backbone_mlo = timm.create_model(**model_config, pretrained=True)
            self.backbone_cc = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(self.backbone_mlo.num_features * 2, 1)
        self._init_weights(self.fc)
        self.loss = self._get_loss()
        self.learning_rate = Last2ImagesCFG.lr

    def forward(self, inputs):
        mlo_image, cc_image = inputs
        mlo_features = self.backbone_mlo(mlo_image)
        cc_features = self.backbone_cc(cc_image)
        features = torch.cat([mlo_features, cc_features], dim=1)
        output = self.fc(features)
        return output

    def configure_optimizers(self):

        optimizer_parameters = self.get_optimizer_parameters()

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=Last2ImagesCFG.lr,
        )

        scheduler = self.get_scheduler(optimizer)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }

    def get_optimizer_parameters(self):
        backbone_params = list(self.backbone_mlo.named_parameters()) + list(self.backbone_cc.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                'weight_decay': Last2ImagesCFG.weight_decay,
                'lr': self.learning_rate * Last2ImagesCFG.backbone_lr_ratio
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * Last2ImagesCFG.backbone_lr_ratio
            },
            {
                'params': self.fc.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate * Last2ImagesCFG.fc_lr_ratio
            }
        ]

        return optimizer_parameters

    def get_scheduler(self, optimizer):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.5
        )
        return scheduler

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, outputs, labels

    def test_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        return outputs

    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_preds = torch.cat([x[1] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()
        all_labels = torch.cat([x[2] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()
        self.log('label_mean', all_labels.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean', (1 / (1 + np.exp(-all_preds))).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean_0.0', (all_preds >= 0.0).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        score, auc, thresh = get_score(all_labels, all_preds)
        self.log("score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("thresh", thresh, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return score

    @staticmethod
    def _get_loss():
        if Last2ImagesCFG.loss_function == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(Last2ImagesCFG.pos_weight))
        elif Last2ImagesCFG.loss_function == "SigmoidFocalLoss":
            return SigmoidFocalLoss(
                gamma=Last2ImagesCFG.focal_loss_gamma,
                alpha=Last2ImagesCFG.focal_loss_alpha,
                sigmoid=True
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()


if __name__ == '__main__':
    model = LitModel()
    print(model)
