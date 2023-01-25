from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.mean_agg.config import MeanAggCFG
from single.mean_agg.data_module import DataModule
from data import load_processed_data
from single.mean_agg.model.lit_module import LitModel
from metrics import calc_oof_score


def inference(seed):
    pl.seed_everything(seed)
    submission_df = load_processed_data.sample_submission(seed=seed)

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
            devices=[0],
            # precision="bf16",
        )

        fold_preds = trainer.predict(
            model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        fold_preds = (torch.concat(fold_preds, axis=0).cpu().detach().float().numpy())

        # 平均を取る前にシグモイドをかける場合〜
        fold_preds = 1 / (1 + np.exp(-fold_preds))

        predictions_list.append(fold_preds)
        if GeneralCFG.debug and fold == 1:
            break

    predictions = np.mean(predictions_list, axis=0)
    # 平均をとった後にシグモイドをかける場合〜
    # predictions = 1 / (1 + np.exp(-predictions))

    predictions = (predictions >= 0.73).astype(int)
    predictions_df = submission_df[["prediction_id"]]
    predictions_df["cancer"] = predictions

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
