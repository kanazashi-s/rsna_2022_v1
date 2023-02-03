from pathlib import Path
import numpy as np
import polars as pol
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG
from single.mean_agg.data_module import DataModule
from data import load_processed_data_pol
from single.mean_agg.model.lit_module import LitModel
from metrics import calc_oof_score


def inference(seed):
    pl.seed_everything(seed)
    submission_df = load_processed_data.sample_submission(seed=seed)
    test_df = load_processed_data_pol.test(seed=seed).with_column(
        pol.lit(0).alias("prediction")
    )

    predictions_list = []
    for fold in GeneralCFG.train_fold:
        input_dir = MeanAggCFG.uploaded_model_dir / f"seed{seed}"
        model = LitModel()

        data_module = DataModule(
            seed=seed,
            fold=fold,
            batch_size=MeanAggCFG.batch_size,
            num_workers=GeneralCFG.num_workers,
        )
        data_module.setup()

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],  # For the Kaggle environment
            # precision="bf16",  # For the Kaggle environment
        )

        fold_preds = trainer.predict(
            model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        fold_preds = (torch.concat(fold_preds, axis=0).cpu().detach().float().numpy())
        fold_preds = 1 / (1 + np.exp(-fold_preds))

        # test_df の prediction に、 fold_preds を fold 数で割って足す
        test_df = test_df.with_column(
            pol.lit(fold_preds).alias("prediction_i")
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
            (pol.col("prediction").mean() >= 0.52).cast(pol.Int32).alias("cancer")
        ),
        on="prediction_id",
        how="left"
    )

    assert predictions_df.get_column("prediction_id").series_equal(submission_df.get_column("prediction_id"))
    assert pol.all(predictions_df.select(
        pol.col("cancer").is_not_null()
    ))

    return predictions_df


def calc_seed_mean(seeds=GeneralCFG.seeds):
    predictions_df_list = []
    for seed in seeds:
        predictions = inference(seed)
        predictions_df_list.append(predictions)

    predictions_seed_mean_df = predictions_df_list[0][["prediction_id"]].copy()
    predictions_seed_mean_df["cancer"] = np.mean(
        [predictions_df["cancer"].values for predictions_df in predictions_df_list],
        axis=0
    )
    predictions_seed_mean_df["cancer"] = (predictions_seed_mean_df["cancer"] >= 0.5).astype(float)
    return predictions_seed_mean_df


if __name__ == "__main__":
    MeanAggCFG.uploaded_model_dir = Path("/workspace", "output", "single", "mean_agg", "baseline_512")
    predictions_seed_mean_df = calc_seed_mean()
    predictions_seed_mean_df.to_csv(MeanAggCFG.uploaded_model_dir / "submission.csv", index=False)
