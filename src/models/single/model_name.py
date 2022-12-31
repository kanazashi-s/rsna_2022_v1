import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from config.general import GeneralCFG
from config.single.model_name import ModelNameCFG
from utils.metrics import get_score


class LitModel(pl.LightningModule):
    def __init__(self, is_kaggle=False, ):
        super().__init__()

        # define pretrained model
        if is_kaggle:
            self.backbone = None
        else:
            self.backbone = None

        self.pool = None
        self.fc = None
        self._init_weights(self.fc)
        self.loss = None

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        feature = self.pool(outputs[0], inputs['attention_mask']).half()
        output = self.fc(feature)
        return output

    def configure_optimizers(self):

        optimizer_parameters = self.get_optimizer_parameters()

        optimizer = AdamW(
            optimizer_parameters,
            **ModelNameCFG.optimizer_params
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
                'weight_decay': weight_decay,
                'lr': self.learning_rate * ModelNameCFG.backbone_lr_ratio
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * ModelNameCFG.backbone_lr_ratio
            },
            {
                'params': self.pool.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate * ModelNameCFG.pool_lr_ratio
            },
            {
                'params': self.fc.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate * ModelNameCFG.fc_lr_ratio
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

    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_preds = torch.cat([x[1] for x in validation_step_outputs], dim=0).cpu().detach().numpy()
        all_labels = torch.cat([x[2] for x in validation_step_outputs], dim=0).cpu().detach().numpy()
        score = get_score(all_labels, all_preds)
        self.log("score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return score

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()


if __name__ == '__main__':
    model = LitModel()
    print(model)
