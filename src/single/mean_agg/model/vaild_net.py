import numpy as np
import torch
import pytorch_lightning as pl
from metrics import get_scores


class ValidNetMixin(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, outputs, labels

    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_preds = torch.cat([x[1] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()
        all_labels = torch.cat([x[2] for x in validation_step_outputs], dim=0).cpu().detach().float().numpy()
        self.log('label_mean', all_labels.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean', (1 / (1 + np.exp(-all_preds))).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean_0.0', (all_preds >= 0.0).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # TODO: validation dataset に合わせて、集約を行う

        scores_dict = get_scores.get(all_labels, all_preds)
        for key, value in scores_dict.items():
            # key の末尾が curve の場合は、 plt.Figure オブジェクトが格納されているため、Figure としてロギングする
            if key.endswith("curve"):
                self.logger.experiment.log_figure(
                    run_id=self.logger.run_id, figure=value, artifact_file=f"{key}_{self.trainer.global_step}.png"
                )
            else:
                self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return

