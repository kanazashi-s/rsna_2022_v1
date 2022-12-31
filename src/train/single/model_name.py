from pathlib import Path
import shutil
import logging
import gc
import numpy as np
import torch
import pytorch_lightning as pl
from config.general import GeneralCFG
from config.single.model_name import ModelNameCFG
from data.data_module import DataModule
from models.single.model_name import LitModel
from evaluation.single.model_name import evaluate
from utils.upload_model import create_dataset_metadata


def train(run_name: str):

    score_list = []

    for seed in GeneralCFG.seeds:
        output_dir = ModelNameCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = DataModule(
                seed=seed,
                fold=fold,
                batch_size=ModelNameCFG.batch_size,
                num_workers=GeneralCFG.num_workers,
            )
            data_module.setup()

            model = LitModel()

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + ModelNameCFG.model_name,
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor="score",
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode="min",
                every_n_train_steps=ModelNameCFG.val_check_interval
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            callbacks = [loss_callback, lr_monitor]

            trainer = pl.Trainer(
                devices=[1],
                accelerator="gpu",
                max_epochs=ModelNameCFG.epochs,
                precision=16,
                amp_backend='apex',
                amp_level='O2',
                gradient_clip_val=ModelNameCFG.max_grad_norm,
                accumulate_grad_batches=ModelNameCFG.accumulate_grad_batches,
                logger=mlflow_logger,
                default_root_dir=output_dir,
                callbacks=callbacks,
                val_check_interval=ModelNameCFG.val_check_interval,
            )
            trainer.fit(model, data_module)
            torch.cuda.empty_cache()
            gc.collect()

        oof_df, score = evaluate(seed)
        mlflow_logger.log_metrics({
            "oof_score": score,
        })

        upload_model.create_dataset_metadata(
            model_name=f"model-name-seed{seed}",
            model_path=output_dir,
        )

        score_list.append(score)

    oof_score_seed_mean = sum(score_list) / len(score_list)
    mlflow_logger.log_metrics({
        "oof_score_seed_mean": oof_score_seed_mean,
    })

    print(f"oof_score_seed_mean: {oof_score_seed_mean}")

    upload_model.create_dataset_metadata(
        model_name=f"model-name-v3-base",
        model_path=ModelNameCFG.output_dir,
    )

    return oof_score_seed_mean


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": ModelNameCFG.model_name,
        "lr": ModelNameCFG.lr,
        "epochs": ModelNameCFG.epochs,
        "pooling": ModelNameCFG.pooling,
        "max_grad_norm": ModelNameCFG.max_grad_norm,
        "accumulate_grad_batches": ModelNameCFG.accumulate_grad_batches,
    }
    return param_dict


if __name__ == "__main__":
    ModelNameCFG.output_dir = Path("/workspace", "output", "single", "model_name")
    oof_score_seed_mean = train(f"baseline")


