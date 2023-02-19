import numpy as np
import polars as pol
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from metrics import get_scores
from data import load_processed_data_pol


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
        all_preds = (1 / (1 + np.exp(-all_preds)))
        self.log('label_mean', all_labels.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred_mean', all_preds.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        valid_fold_num = self.trainer.datamodule.fold
        seed_num = self.trainer.datamodule.seed
        whole_df = load_processed_data_pol.train(seed=seed_num)
        valid_df = whole_df.filter(
            pol.col('fold') == valid_fold_num
        ).select([
            pol.col("prediction_id"),
            pol.col("cancer"),
            pol.Series(name="all_preds", values=[0] if self.trainer.sanity_checking else all_preds.reshape(-1)),
        ])

        if not self.trainer.sanity_checking:
            assert valid_df.get_column("cancer").series_equal(pol.Series(name="cancer", values=all_labels.reshape(-1)))

        agg_df = load_processed_data_pol.sample_oof(seed=seed_num).filter(
            pol.col('fold') == valid_fold_num
        ).select(
            pol.col("prediction_id")
        ).join(
            valid_df.groupby("prediction_id").agg([
                pol.col("all_preds").mean().alias("all_preds"),
                pol.col("cancer").first().alias("cancer")
            ]),
            on="prediction_id",
            how="left"
        )

        all_labels = agg_df.select(pol.col("cancer")).to_numpy().flatten()
        all_preds = agg_df.select(pol.col("all_preds")).to_numpy().flatten()

        scores_dict = get_scores.get(all_labels, all_preds, is_sigmoid=True)
        for key, value in scores_dict.items():
            # key の末尾が curve の場合は、 plt.Figure オブジェクトが格納されているため、Figure としてロギングする
            if key.endswith("curve"):
                self.logger.experiment.log_figure(
                    run_id=self.logger.run_id, figure=value, artifact_file=f"{key}_{self.trainer.global_step}.png"
                )
            else:
                self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return

