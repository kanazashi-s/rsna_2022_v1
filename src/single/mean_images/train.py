from pathlib import Path
import shutil
import logging
import gc
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.mean_images.config import MeanImagesCFG
from single.mean_images.data_module import DataModule
from single.mean_images.lit_module import LitModel
from single.mean_images.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def train(run_name: str, seed_list=None, device_idx=0):

    score_list = []
    if seed_list is None:
        seed_list = GeneralCFG.seeds

    for seed in seed_list:
        output_dir = MeanImagesCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = DataModule(
                seed=seed,
                fold=fold,
                batch_size=MeanImagesCFG.batch_size,
                num_workers=GeneralCFG.num_workers,
            )
            data_module.setup()

            model = LitModel()

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + MeanImagesCFG.model_name,
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            val_check_interval = len(data_module.train_dataloader()) // 5

            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor=MeanImagesCFG.monitor_metric,
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode=MeanImagesCFG.monitor_mode,
                every_n_train_steps=val_check_interval,
                verbose=True,
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            callbacks = [loss_callback, lr_monitor]

            trainer = pl.Trainer(
                devices=[device_idx],
                accelerator="gpu",
                max_epochs=MeanImagesCFG.epochs,
                precision="bf16",
                amp_backend='native',
                gradient_clip_val=MeanImagesCFG.max_grad_norm,
                accumulate_grad_batches=MeanImagesCFG.accumulate_grad_batches,
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
        model_path=MeanImagesCFG.output_dir,
    )

    return oof_score_seed_mean


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": MeanImagesCFG.model_name,
        "lr": MeanImagesCFG.lr,
        "batch_size": MeanImagesCFG.batch_size,
        "epochs": MeanImagesCFG.epochs,
        "max_grad_norm": MeanImagesCFG.max_grad_norm,
        "accumulate_grad_batches": MeanImagesCFG.accumulate_grad_batches,
        "loss_function": MeanImagesCFG.loss_function,
        "pos_weight": MeanImagesCFG.pos_weight,
        "focal_loss_alpha": MeanImagesCFG.focal_loss_alpha,
        "focal_loss_gamma": MeanImagesCFG.focal_loss_gamma,
        "monitor_metric": MeanImagesCFG.monitor_metric,
        "monitor_mode": MeanImagesCFG.monitor_mode,
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

    # MeanImagesCFG.model_name = "efficientnetv2_rw_m"
    # MeanImagesCFG.batch_size = 4
    # MeanImagesCFG.accumulate_grad_batches = 16
    # MeanImagesCFG.output_dir = output_dir = Path("/workspace", "output", "single", "last_2_images", "efficientnetv2_rw_m_mean")
    # oof_score_seed_mean = train(f"last_2_images_efficientnetv2_rw_m_mean", seed_list=[42], device_idx=0)

    # MeanImagesCFG.model_name = "efficientnetv2_rw_m"
    # MeanImagesCFG.batch_size = 8
    # MeanImagesCFG.accumulate_grad_batches = 32
    # MeanImagesCFG.sampler = None
    # MeanImagesCFG.loss_function = "MacroSoftF1Loss"
    # MeanImagesCFG.output_dir = Path("/workspace", "output", "single", "last_2_images", "efficientnetv2_rw_m_mean_f1_loss")
    # oof_score_seed_mean = train(f"last_2_images_efficientnetv2_rw_m_mean_f1_loss", seed_list=[42], device_idx=1)

    # MeanImagesCFG.loss_function = "SigmoidFocalLoss"
    # MeanImagesCFG.focal_loss_alpha = 50.0
    # MeanImagesCFG.focal_loss_gamma = 2.0
    # MeanImagesCFG.output_dir = Path("/workspace", "output", "single", "max_images", "efficientnetv2_rw_m_mean_focal")
    # oof_score_seed_mean = train(f"last_2_images_efficientnetv2_rw_m_mean_focal", seed_list=[42], device_idx=1)

    MeanImagesCFG.output_dir = Path("/workspace", "output", "single", "mean_images", "imbalance_focal")
    MeanImagesCFG.loss_function = "SigmoidFocalLoss"
    MeanImagesCFG.focal_loss_alpha = 1.0
    MeanImagesCFG.focal_loss_gamma = 2.0
    MeanImagesCFG.sampler = "ImbalancedDatasetSampler"
    MeanImagesCFG.batch_size = 8
    MeanImagesCFG.accumulate_grad_batches = 32
    GeneralCFG.seeds = [42]
    oof_score_seed_mean = train(
        f"mean-images-imbalance-focal",
        seed_list=GeneralCFG.seeds,
        device_idx=0
    )
