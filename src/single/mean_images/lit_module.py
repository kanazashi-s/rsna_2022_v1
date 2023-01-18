import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from transformers import get_cosine_schedule_with_warmup
from cfg.general import GeneralCFG
from single.mean_images.config import MeanImagesCFG
from utils.model_utils.focal_loss import SigmoidFocalLoss
from utils.model_utils.macro_soft_f1 import MacroSoftF1Loss
from metrics import get_scores
from single.mean_images.postprocessing import agg_by_prediction_id


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # define pretrained model
        model_config = {
            "model_name": MeanImagesCFG.model_name,
            "num_classes": 0,  # to use feature extractor,
            "in_chans": 1,
            "drop_rate": MeanImagesCFG.drop_rate,
            "drop_path_rate": MeanImagesCFG.drop_path_rate,
        }
        if GeneralCFG.is_kaggle:
            self.backbone = timm.create_model(**model_config, pretrained=False)
        else:
            self.backbone = timm.create_model(**model_config, pretrained=True)

        self.fc = nn.Linear(self.backbone.num_features, 1)
        self._init_weights(self.fc)
        self.loss = self._get_loss()
        self.learning_rate = MeanImagesCFG.lr

    def forward(self, image):
        features = self.backbone(image)
        output = self.fc(features)
        return output

    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_parameters()

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=MeanImagesCFG.lr,
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
        backbone_params = list(self.backbone.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                'weight_decay': MeanImagesCFG.weight_decay,
                'lr': self.learning_rate * MeanImagesCFG.backbone_lr_ratio
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * MeanImagesCFG.backbone_lr_ratio
            },
            {
                'params': self.fc.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate * MeanImagesCFG.fc_lr_ratio
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
        inputs = batch[0]
        outputs = self.forward(inputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = self.forward(inputs)
        return outputs

    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_preds = torch.cat([x[1] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()
        all_labels = torch.cat([x[2] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()

        valid_df = self.trainer.datamodule.valid_df[['prediction_id', 'cancer']].copy()

        if self.trainer.sanity_checking:
            valid_df = valid_df.iloc[:len(all_preds)]
        assert np.array_equal(valid_df['cancer'].values, all_labels.flatten().astype(int))

        max_predictions, max_labels = agg_by_prediction_id(valid_df, all_preds)

        self.log('label_mean', all_labels.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean', (1 / (1 + np.exp(-all_preds))).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        score, auc, thresh = get_scores.get(max_labels, max_predictions)

        self.log('pred_mean_thresh', (all_preds >= thresh).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("thresh", thresh, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return score

    @staticmethod
    def _get_loss():
        if MeanImagesCFG.loss_function == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(MeanImagesCFG.pos_weight))
        elif MeanImagesCFG.loss_function == "SigmoidFocalLoss":
            return SigmoidFocalLoss(
                gamma=MeanImagesCFG.focal_loss_gamma,
                alpha=MeanImagesCFG.focal_loss_alpha,
                sigmoid=True
            )
        elif MeanImagesCFG.loss_function == "MacroSoftF1Loss":
            return MacroSoftF1Loss(
                consider_true_negative=False,
                sigmoid_is_applied_to_input=False
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
