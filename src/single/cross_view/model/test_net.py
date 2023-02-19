import pytorch_lightning as pl


class TestNetMixin(pl.LightningModule):
    def test_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        return outputs
