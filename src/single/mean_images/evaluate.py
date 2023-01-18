from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.mean_images.config import MeanImagesCFG
from single.mean_images.data_module import DataModule
from data import load_processed_data
from single.mean_images.lit_module import LitModel
from single.mean_images.postprocessing import agg_by_prediction_id
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
        input_dir = MeanImagesCFG.output_dir / f"seed{seed}"
        model = LitModel()
        data_module = DataModule(
            seed=seed,
            fold=fold,
            batch_size=MeanImagesCFG.batch_size,
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

        max_predictions, max_labels = agg_by_prediction_id(
            data_module.valid_df,
            fold_i_val_pred,
        )

        assert np.array_equal(
            max_labels,
            load_processed_data.train(seed=seed).drop_duplicates("prediction_id").reset_index().loc[
                fold_idx, GeneralCFG.target_col
            ].values
        )

        oof_df.loc[fold_idx, GeneralCFG.target_col] = max_predictions

        if debug and fold == len(GeneralCFG.train_fold) - 1:
            oof_df = oof_df.loc[oof_df.loc[:, GeneralCFG.target_col].notnull()].reset_index(drop=True)
            break

    oof_df.to_csv(MeanImagesCFG.output_dir / "oof.csv", index=False)
    score, auc, thresh, fold_scores, fold_aucs = calc_oof_score.calc(oof_df, is_debug=debug, seed=seed)
    return oof_df, score, auc, thresh, fold_scores, fold_aucs


if __name__ == "__main__":
    MeanImagesCFG.loss_function = "SigmoidFocalLoss"
    MeanImagesCFG.output_dir = Path("/workspace", "output", "single", "last_2_images", "efficientnetv2_rw_m_mean_focal")
    oof_df, score, auc, thresh, fold_scores, fold_aucs = evaluate(seed=42, device_idx=0)
    print(score)
    print(auc)
    print(thresh)
    print(fold_scores)
    print(fold_aucs)
    print(oof_df)
