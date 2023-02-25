import torch
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from single.cross_view.config import CrossViewCFG


class OptimizersMixin(pl.LightningModule):
    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_parameters()

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=CrossViewCFG.lr,
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
        init_params = list(self.cvam_stage2.named_parameters()) + list(self.mlp.named_parameters())
        backbone_params = list(
            self.stage1_backbone.named_parameters())\
            + list(self.stage2_backbone.named_parameters())\
            + list(self.stage0_backbone.named_parameters()
        )

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in init_params if not any(nd in n for nd in no_decay)],
                'weight_decay': CrossViewCFG.weight_decay,
                'lr': self.learning_rate
            },
            {
                'params': [p for n, p in init_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate
            },
            {
                'params': [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                'weight_decay': CrossViewCFG.weight_decay,
                'lr': self.learning_rate * CrossViewCFG.backbone_lr_factor
            },
            {
                'params': [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * CrossViewCFG.backbone_lr_factor
            },
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
