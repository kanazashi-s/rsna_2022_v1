import torch
import pytorch_lightning as pl
from single.mean_agg.config import MeanAggCFG


class PredictNetMixin(pl.LightningModule):
    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)

        if MeanAggCFG.use_tta:
            # 画像を左右反転して、再度予測を実施
            inputs_flip = inputs.flip(dims=[3])
            outputs_flip = self.forward(inputs_flip)

            # 2つの予測値を、それぞれシグモイド関数にかけ、平均を取る
            outputs = (1 / (1 + torch.exp(-outputs))) + (1 / (1 + torch.exp(-outputs_flip)))
            outputs = outputs / 2

            # シグモイド関数の逆変換
            outputs = torch.log(outputs / (1 - outputs))

        return outputs
