from pathlib import Path
import shutil
import logging
import gc
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.last_2_images.config import Last2ImagesCFG
from single.last_2_images.data_module import DataModule
from single.last_2_images.lit_module import LitModel
from single.last_2_images.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def train(run_name: str, seed_list=None, device_idx=0):

    score_list = []
    if seed_list is None:
        seed_list = GeneralCFG.seeds

    for seed in seed_list:
        output_dir = Last2ImagesCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = DataModule(
                seed=seed,
                fold=fold,
                batch_size=Last2ImagesCFG.batch_size,
                num_workers=GeneralCFG.num_workers,
            )
            data_module.setup()

            model = LitModel()

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + Last2ImagesCFG.model_name,
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            val_check_interval = len(data_module.train_dataloader()) // 5

            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor=Last2ImagesCFG.monitor_metric,
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode=Last2ImagesCFG.monitor_mode,
                every_n_train_steps=val_check_interval,
                verbose=True,
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            callbacks = [loss_callback, lr_monitor]

            trainer = pl.Trainer(
                devices=[device_idx],
                accelerator="gpu",
                max_epochs=Last2ImagesCFG.epochs,
                precision="bf16",
                amp_backend='native',
                gradient_clip_val=Last2ImagesCFG.max_grad_norm,
                accumulate_grad_batches=Last2ImagesCFG.accumulate_grad_batches,
                logger=mlflow_logger,
                default_root_dir=output_dir,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
            )
            trainer.fit(model, data_module)
            torch.cuda.empty_cache()
            gc.collect()

        oof_df, score, auc, thresh, fold_scores, fold_aucs = evaluate(seed, device_idx)
        mlflow_logger.log_metrics(get_metric_dict(score, auc, thresh, fold_scores, fold_aucs))

        create_dataset_metadata(
            model_name=f"last-2-images-seed{seed}",
            model_path=output_dir,
        )

        score_list.append(score)

    oof_score_seed_mean = sum(score_list) / len(score_list)
    mlflow_logger.log_metrics({
        "oof_score_seed_mean": oof_score_seed_mean,
    })

    print(f"oof_score_seed_mean: {oof_score_seed_mean}")

    create_dataset_metadata(
        model_name=f"last-2-images",
        model_path=Last2ImagesCFG.output_dir,
    )

    return oof_score_seed_mean


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": Last2ImagesCFG.model_name,
        "lr": Last2ImagesCFG.lr,
        "batch_size": Last2ImagesCFG.batch_size,
        "epochs": Last2ImagesCFG.epochs,
        "max_grad_norm": Last2ImagesCFG.max_grad_norm,
        "accumulate_grad_batches": Last2ImagesCFG.accumulate_grad_batches,
        "loss_function": Last2ImagesCFG.loss_function,
        "pos_weight": Last2ImagesCFG.pos_weight,
        "focal_loss_alpha": Last2ImagesCFG.focal_loss_alpha,
        "focal_loss_gamma": Last2ImagesCFG.focal_loss_gamma,
        "monitor_metric": Last2ImagesCFG.monitor_metric,
        "monitor_mode": Last2ImagesCFG.monitor_mode,
    }
    return param_dict


def get_metric_dict(score, auc, thresh, fold_scores, fold_aucs):
    metric_dict = {
        "oof_score": score,
        "oof_auc": auc,
        "oof_thresh": thresh,
    }
    for i, (fold_score, fold_auc) in enumerate(zip(fold_scores, fold_aucs)):
        metric_dict[f"fold{i}_score"] = fold_score
        metric_dict[f"fold{i}_auc"] = fold_auc
    return metric_dict


if __name__ == "__main__":
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_1024"
    # GeneralCFG.image_size = 1024
    # Last2ImagesCFG.model_name = "resnetrs152"
    # Last2ImagesCFG.batch_size = 4
    # Last2ImagesCFG.accumulate_grad_batches = 16
    # Last2ImagesCFG.lr = 6e-4
    # Last2ImagesCFG.output_dir = output_dir = Path("/workspace", "output", "single", "last_2_images", "baseline")
    # oof_score_seed_mean = train(f"last_2_images", seed_list=[42], device_idx=0)

    Last2ImagesCFG.model_name = "efficientnetv2_rw_m"
    Last2ImagesCFG.batch_size = 4
    Last2ImagesCFG.accumulate_grad_batches = 16
    Last2ImagesCFG.output_dir = output_dir = Path("/workspace", "output", "single", "last_2_images", "efficientnetv2_rw_m_lr1e-4")
    oof_score_seed_mean = train(f"last_2_images_efficientnetv2_rw_m_lr1e-4", seed_list=[42], device_idx=0)

