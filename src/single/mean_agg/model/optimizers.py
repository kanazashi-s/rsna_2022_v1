import torch
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from single.mean_agg.config import MeanAggCFG


class OptimizersMixin(pl.LightningModule):
    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_parameters()

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=MeanAggCFG.lr,
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
        backbone_params = self.backbone.named_parameters()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                'weight_decay': MeanAggCFG.weight_decay,
                'lr': self.learning_rate * MeanAggCFG.backbone_lr_ratio
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * MeanAggCFG.backbone_lr_ratio
            },
            {
                'params': self.fc.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate * MeanAggCFG.fc_lr_ratio
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
