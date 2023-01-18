import pytorch_lightning as pl


class PredictNetMixin(pl.LightningModule):
    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        return outputs
