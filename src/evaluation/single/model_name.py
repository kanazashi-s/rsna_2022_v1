import numpy as np
import torch
import pytorch_lightning as pl
from config.general import GeneralCFG
from config.single.model_name import ModelNameCFG
from data.data_module import DataModule
from data import load_processed_data
from models.single.model_name import LitModel
from utils import metrics


def evaluate(seed):
    """
    学習済みモデルの評価
    """
    pl.seed_everything(seed)
    debug = GeneralCFG.debug

    oof_df = load_processed_data.train(seed=seed)[["id_col", "fold"] + [GeneralCFG.target_col]]
    oof_df[GeneralCFG.target_col] = np.nan

    if debug:
        oof_df = oof_df.loc[oof_df["fold"].isin(GeneralCFG.train_fold), :].reset_index(drop=True)

    for fold in GeneralCFG.train_fold:
        input_dir = ModelNameCFG.output_dir / f"seed{seed}"
        model = LitModel()
        data_module = DataModule(
            seed=seed,
            fold=fold,
            batch_size=ModelNameCFG.batch_size,
            num_workers=GeneralCFG.num_workers,
        )
        data_module.setup()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[1],
            precision=16,
        )
        fold_i_val_pred = trainer.predict(
            model,
            dataloaders=data_module,
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        fold_i_val_pred = (torch.concat(fold_i_val_pred, axis=0).numpy())

        fold_idx = oof_df.loc[oof_df["fold"] == fold].index
        if debug:
            fold_idx = fold_idx[:GeneralCFG.num_use_data]

        oof_df.loc[fold_idx, GeneralCFG.target_col] = fold_i_val_pred

        if debug and fold == len(GeneralCFG.train_fold) - 1:
            oof_df = oof_df.loc[oof_df.loc[:, GeneralCFG.target_col].notnull()].reset_index(drop=True)
            break

    oof_df.to_csv(ModelNameCFG.output_dir / "oof.csv", index=False)
    score = metrics.get_oof_score(oof_df, is_debug=debug, seed=seed)
    return oof_df, score


if __name__ == "__main__":
    from pathlib import Path
    GeneralCFG.data_version = "vanilla"
    ModelNameCFG.output_dir = Path("/workspace", "output", "single", "model_name")
    oof_df, score = evaluate(seed=42)
    print(score)
    print(oof_df)
