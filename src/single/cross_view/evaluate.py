from pathlib import Path
import numpy as np
import polars as pol
import torch
import pytorch_lightning as pl
import ttach as tta
from cfg.general import GeneralCFG
from single.cross_view.config import CrossViewCFG
from single.cross_view.data_module import DataModule
from data import load_processed_data_pol
from single.cross_view.model.lit_module import LitModel
from metrics import calc_oof_score_pol


def evaluate(seed, device_idx=0):
    """
    Evaluation
    """
    pl.seed_everything(seed)
    debug = GeneralCFG.debug

    if debug:
        oof_df = load_processed_data_pol.debug_sample_oof(seed=seed)
        train_df = load_processed_data_pol.debug_train(seed=seed)
    else:
        oof_df = load_processed_data_pol.sample_oof(seed=seed)
        train_df = load_processed_data_pol.train(seed=seed)

    train_df = train_df.filter(
        pol.col("fold").is_in(GeneralCFG.train_fold)
    ).with_column(
        pol.lit(0).alias("prediction")
    )

    for fold in GeneralCFG.train_fold:
        input_dir = CrossViewCFG.output_dir / f"seed{seed}"
        model = LitModel()
        data_module = DataModule(
            seed=seed,
            fold=fold,
            num_workers=GeneralCFG.num_workers,
        )
        data_module.setup()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[device_idx],
            precision="bf16",
        )
        fold_i_val_pred = trainer.predict(
            model,
            dataloaders=data_module,
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        fold_i_val_pred = (torch.concat(fold_i_val_pred, axis=0).cpu().detach().float().numpy())
        fold_i_val_pred = 1 / (1 + np.exp(-fold_i_val_pred))

        train_df = train_df.join(
            train_df.filter(
                pol.col("fold") == fold
            ).select(
                pol.col("image_id"),
            ).with_column(
                pol.lit(fold_i_val_pred.reshape(-1)).alias("prediction_i")
            ),
            on="image_id",
            how="left"
        ).select([
            pol.col("prediction_id"),
            pol.col("image_id"),
            pol.col("fold"),
            (pol.col("prediction") + pol.col("prediction_i").fill_null(pol.lit(0))).alias("prediction"),
            pol.col("cancer")
        ])

    train_df.write_csv(CrossViewCFG.output_dir / "oof_before_agg.csv")

    oof_df = oof_df.filter(
        pol.col("fold").is_in(GeneralCFG.train_fold)
    ).select([
        pol.col("prediction_id"),
        pol.col("fold"),
    ]).join(
        train_df.groupby("prediction_id").agg([
            pol.col("prediction").mean().alias("prediction"),
            pol.col("cancer").first().alias("cancer")
        ]),
        on="prediction_id",
        how="left"
    )

    oof_df.write_csv(CrossViewCFG.output_dir / "oof.csv")
    whole_metrics, metrics_by_folds, metrics_each_fold = calc_oof_score_pol.calc(oof_df, is_debug=debug, seed=seed, is_sigmoid=True)
    return whole_metrics, metrics_by_folds, metrics_each_fold


if __name__ == "__main__":
    CrossViewCFG.output_dir = Path("/workspace", "output", "single", "cross_view", "1536_ker_swa_smooth")
    # GeneralCFG.image_size = 1536
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "1536_ker_png"
    # GeneralCFG.num_workers = 2
    # CrossViewCFG.model_name = "efficientnetv2_rw_s"
    # whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed=42, device_idx=1)
    # print(whole_metrics)

    oof_df = pol.read_csv(CrossViewCFG.output_dir / "oof.csv")
    whole_metrics, metrics_by_folds, metrics_each_fold = calc_oof_score_pol.calc(oof_df, is_debug=False, seed=42, is_sigmoid=True)
    print(whole_metrics)

