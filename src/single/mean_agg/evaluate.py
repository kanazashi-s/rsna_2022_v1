from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.two_view_concat.config import TwoViewConcatCFG
from single.two_view_concat.data_module import DataModule
from data import load_processed_data
from single.two_view_concat.model.lit_module import LitModel
from metrics import calc_oof_score


def evaluate(seed, device_idx=0):
    """
    学習済みモデルの評価
    """
    pl.seed_everything(seed)
    debug = GeneralCFG.debug

    oof_df = load_processed_data.sample_oof(seed=seed)
    oof_df[GeneralCFG.target_col] = np.nan

    if debug:
        oof_df = oof_df.loc[oof_df["fold"].isin(GeneralCFG.train_fold), :].reset_index(drop=True)

    for fold in GeneralCFG.train_fold:
        input_dir = TwoViewConcatCFG.output_dir / f"seed{seed}"
        model = LitModel()
        data_module = DataModule(
            seed=seed,
            fold=fold,
            batch_size=TwoViewConcatCFG.batch_size,
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

        fold_idx = oof_df.loc[oof_df["fold"] == fold].index
        if debug:
            fold_idx = fold_idx[:GeneralCFG.num_use_data]

        oof_df.loc[fold_idx, GeneralCFG.target_col] = fold_i_val_pred

        if debug and fold == len(GeneralCFG.train_fold) - 1:
            oof_df = oof_df.loc[oof_df.loc[:, GeneralCFG.target_col].notnull()].reset_index(drop=True)
            break

    oof_df.to_csv(TwoViewConcatCFG.output_dir / "oof.csv", index=False)
    whole_metrics, metrics_by_folds, metrics_each_fold = calc_oof_score.calc(oof_df, is_debug=debug, seed=seed)
    return whole_metrics, metrics_by_folds, metrics_each_fold


if __name__ == "__main__":
    TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "baseline_1024")
    GeneralCFG.image_size = 1024
    GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_1024"
    whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed=42, device_idx=0)
    print(metrics_by_folds)

