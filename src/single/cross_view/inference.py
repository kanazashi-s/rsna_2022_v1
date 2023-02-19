import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

from pathlib import Path
import numpy as np
import polars as pol
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
import torch_tensorrt
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG
from data import load_processed_data_pol
from single.mean_agg.dataset import TestDataset
from single.mean_agg.transforms import get_transforms
from single.mean_agg.model.lit_module import LitModel
from single.mean_agg.model.litmodule_to_trt import MyModel
from metrics import calc_oof_score
from memory_profiler import profile


@profile
def inference(seed):
    pl.seed_everything(seed)
    submission_df = load_processed_data_pol.sample_submission(seed=seed)

    if GeneralCFG.debug:
        test_df = load_processed_data_pol.train(seed=seed)[:1000]
        GeneralCFG.train_fold = [0, 1, 2, 3, 4]
        GeneralCFG.num_workers = 2
    else:
        test_df = load_processed_data_pol.test(seed=seed)

    test_df = test_df.with_column(
        pol.lit(0).alias("prediction")
    ).with_column(
        (pol.col("patient_id").cast(pol.Utf8) + "_" + pol.col("image_id").cast(pol.Utf8) + ".png")
        .alias("image_filename")
    )

    transforms_no_aug = get_transforms(augment=False)
    test_dataset = TestDataset(test_df, transforms_no_aug, is_inference=True)

    for fold in GeneralCFG.train_fold:
        input_dir = MeanAggCFG.trt_model_dir / f"seed{seed}"
        trt_ts_module = torch.jit.load(input_dir / f"trt_fold{fold}.ts")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=MeanAggCFG.batch_size,
            shuffle=False,
            num_workers=GeneralCFG.num_workers,
            pin_memory=True,
        )

        # torch のモデルを使って、 test_df の prediction を予測
        fold_preds = []
        for batch in test_dataloader:
            batch = batch.cuda().half()
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    if MeanAggCFG.use_tta:
                        pred1 = trt_ts_module(batch)
                        pred2 = trt_ts_module(batch.flip(dims=[3]))
                        # 2つの予測値を、それぞれシグモイド関数にかけ、平均を取る
                        pred = (1 / (1 + torch.exp(-pred1))) + (1 / (1 + torch.exp(-pred2)))
                        pred = pred / 2
                        # シグモイド関数の逆変換
                        pred = torch.log(pred / (1 - pred))
                        fold_preds.append(pred)
                    else:
                        fold_preds.append(trt_ts_module(batch))

        fold_preds = (torch.concat(fold_preds, axis=0).cpu().detach().float().numpy())
        fold_preds = 1 / (1 + np.exp(-fold_preds))

        # test_df の prediction に、 fold_preds を fold 数で割って足す
        test_df = test_df.with_column(
            pol.lit(fold_preds.reshape(-1)).alias("prediction_i")
        ).select([
            pol.col("prediction_id"),
            (pol.col("prediction") + pol.col("prediction_i") / len(GeneralCFG.train_fold)).alias("prediction")
        ])

        if GeneralCFG.debug and fold == 1:
            break

    # test_df の prediction_id ごとに、 prediction の平均を取る
    predictions_df = submission_df.select([
        pol.col("prediction_id"),
    ]).join(
        test_df.groupby("prediction_id").agg(
            (pol.col("prediction").mean() >= 0.39).cast(pol.Int32).alias("cancer")
        ),
        on="prediction_id",
        how="left"
    )

    if not GeneralCFG.debug:
        assert predictions_df.get_column("prediction_id").series_equal(submission_df.get_column("prediction_id"))
        assert np.all(predictions_df.select(pol.col("cancer").is_not_null()).to_numpy())

    return predictions_df


def calc_seed_mean(seeds=GeneralCFG.seeds, num_workers=None, batch_size=None):
    if num_workers is not None:
        GeneralCFG.num_workers = num_workers
    if batch_size is not None:
        MeanAggCFG.batch_size = batch_size

    predictions_df_list = []
    for seed in seeds:
        predictions = inference(seed)
        predictions_df_list.append(predictions)

    predictions_seed_mean_df = predictions_df_list[0].select(pol.col("prediction_id")).with_column(
        pol.lit(np.mean(
            [predictions_df.get_column("cancer").to_numpy() for predictions_df in predictions_df_list],
            axis=0
        )).alias("cancer")
    ).with_column(
        (pol.col("cancer") >= 0.5).cast(pol.Int32).alias("cancer")
    )

    return predictions_seed_mean_df


if __name__ == "__main__":
    MeanAggCFG.trt_model_dir = Path("/workspace", "output", "single", "mean_agg", "1536_ker_swa")
    GeneralCFG.num_workers = 0
    predictions_seed_mean_df = calc_seed_mean()
    predictions_seed_mean_df.write_csv(MeanAggCFG.uploaded_model_dir / "submission.csv")
