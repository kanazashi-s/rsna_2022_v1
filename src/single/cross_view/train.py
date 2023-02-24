from pathlib import Path
import shutil
import gc
from collections import defaultdict
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.cross_view.config import CrossViewCFG
from single.cross_view.data_module import DataModule
from single.cross_view.model.lit_module import LitModel
from single.cross_view.evaluate import evaluate
from single.cross_view.model.litmodule_to_trt import litmodule_to_trt
from utils.upload_model import create_dataset_metadata


def train(run_name: str, seed_list=None, device_idx=0):

    metrics_seed_dict = defaultdict(list)
    if seed_list is None:
        seed_list = GeneralCFG.seeds

    for seed in seed_list:
        if GeneralCFG.debug:
            CrossViewCFG.output_dir = CrossViewCFG.output_dir.with_name(f"{CrossViewCFG.output_dir.name}_debug")
        output_dir = CrossViewCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = DataModule(
                seed=seed,
                fold=fold,
                num_workers=GeneralCFG.num_workers,
            )
            data_module.setup()

            model = LitModel()

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + f"cross_view_{CrossViewCFG.model_name}",
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            val_check_interval = len(data_module.train_dataloader()) // CrossViewCFG.val_check_per_epoch

            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor=CrossViewCFG.monitor_metric,
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode=CrossViewCFG.monitor_mode,
                verbose=True,
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            swa_callback = pl.callbacks.StochasticWeightAveraging(
                swa_lrs=1e-5,
                swa_epoch_start=20,
                annealing_epochs=0,
            )

            callbacks = [loss_callback, lr_monitor, swa_callback]

            trainer = pl.Trainer(
                profiler="simple",
                devices=[device_idx],
                accelerator="gpu",
                # strategy="ddp",
                max_epochs=CrossViewCFG.epochs,
                precision="bf16",
                amp_backend='native',
                gradient_clip_val=CrossViewCFG.max_grad_norm,
                accumulate_grad_batches=CrossViewCFG.accumulate_grad_batches,
                logger=mlflow_logger,
                default_root_dir=output_dir,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
            )
            trainer.fit(model, data_module)
            torch.cuda.empty_cache()
            gc.collect()

        whole_metrics, metrics_by_folds, metrics_each_fold = evaluate(seed, device_idx)
        log_all_metrics(mlflow_logger, whole_metrics, metrics_by_folds, metrics_each_fold)

        create_dataset_metadata(
            model_name=f"{CrossViewCFG.upload_name}-seed{seed}",
            model_path=output_dir,
        )

        metrics_seed_dict["whole_best_pfbeta"].append(whole_metrics["best_pfbeta"])
        metrics_seed_dict["whole_auc_roc"].append(whole_metrics["auc_roc"])
        metrics_seed_dict["whole_auc_pr"].append(whole_metrics["auc_pr"])
        metrics_seed_dict["by_folds_best_pfbeta"].append(metrics_by_folds["pfbeta"])
        metrics_seed_dict["by_folds_auc_roc"].append(metrics_by_folds["auc_roc"])
        metrics_seed_dict["by_folds_auc_pr"].append(metrics_by_folds["auc_pr"])

    # metrics_seed_dict のすべての要素に対し、平均を計算し、ロギングする
    metrics_seed_mean_dict = {}
    for key, value in metrics_seed_dict.items():
        metrics_seed_mean_dict[f"seed_mean_{key}"] = np.mean(value)
    mlflow_logger.log_metrics(metrics_seed_mean_dict)

    create_dataset_metadata(
        model_name=CrossViewCFG.upload_name,
        model_path=CrossViewCFG.output_dir,
    )

    return metrics_seed_mean_dict


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": CrossViewCFG.model_name,
        "lr": CrossViewCFG.lr,
        "epochs": CrossViewCFG.epochs,
        "max_grad_norm": CrossViewCFG.max_grad_norm,
        "accumulate_grad_batches": CrossViewCFG.accumulate_grad_batches,
        "loss_function": CrossViewCFG.loss_function,
        "pos_weight": CrossViewCFG.pos_weight,
        "focal_loss_alpha": CrossViewCFG.focal_loss_alpha,
        "focal_loss_gamma": CrossViewCFG.focal_loss_gamma,
        "monitor_metric": CrossViewCFG.monitor_metric,
        "monitor_mode": CrossViewCFG.monitor_mode,
    }
    return param_dict


def log_all_metrics(mlflow_logger, whole_metrics, metrics_by_folds, metrics_each_fold):
    # log num metrics
    num_metrics_dict = {
        "whole_pfbeta": whole_metrics["best_pfbeta"],
        "whole_best_threshold": whole_metrics["best_thresh_pfbeta"],
        "whole_auc_roc": whole_metrics["auc_roc"],
        "whole_auc_pr": whole_metrics["auc_pr"],
        "whole_TP": whole_metrics["TP"],
        "whole_FP": whole_metrics["FP"],
        "whole_FN": whole_metrics["FN"],
        "whole_TN": whole_metrics["TN"],
        "by_fold_pfbeta": metrics_by_folds["pfbeta"],
        "by_fold_auc_roc": metrics_by_folds["auc_roc"],
        "by_fold_auc_pr": metrics_by_folds["auc_pr"],
        "by_fold_TP": metrics_by_folds["TP"],
        "by_fold_FP": metrics_by_folds["FP"],
        "by_fold_FN": metrics_by_folds["FN"],
        "by_fold_TN": metrics_by_folds["TN"],
    }
    for i, metrics_one_fold in enumerate(metrics_each_fold):
        num_metrics_dict[f"fold{i}_pfbeta"] = metrics_one_fold["best_pfbeta"]
        num_metrics_dict[f"fold{i}_best_threshold"] = metrics_one_fold["best_thresh_pfbeta"]
        num_metrics_dict[f"fold{i}_auc_roc"] = metrics_one_fold["auc_roc"]
        num_metrics_dict[f"fold{i}_auc_pr"] = metrics_one_fold["auc_pr"]
        num_metrics_dict[f"fold{i}_TP"] = metrics_one_fold["TP"]
        num_metrics_dict[f"fold{i}_FP"] = metrics_one_fold["FP"]
        num_metrics_dict[f"fold{i}_FN"] = metrics_one_fold["FN"]
        num_metrics_dict[f"fold{i}_TN"] = metrics_one_fold["TN"]

    mlflow_logger.log_metrics(num_metrics_dict)

    # log fig metrics
    run_id = mlflow_logger.run_id
    for key, value in whole_metrics.items():
        if key.endswith("curve"):
            mlflow_logger.experiment.log_figure(run_id, value, f"whole_{key}.png")
    for i, metrics_one_fold in enumerate(metrics_each_fold):
        for key, value in metrics_one_fold.items():
            if key.endswith("curve"):
                mlflow_logger.experiment.log_figure(run_id, value, f"each_fold_{i}_{key}.png")


if __name__ == "__main__":
    if GeneralCFG.debug:
        GeneralCFG.num_workers = 0
    else:
        GeneralCFG.num_workers = 2

    CrossViewCFG.accumulate_grad_batches = 24
    CrossViewCFG.output_dir = Path("/workspace", "output", "single", "cross_view", "1536_ker_swa_smooth")
    GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "1536_ker_png"
    CrossViewCFG.model_name = "efficientnetv2_rw_s"
    train(f"cross_view_1536_ker_baseline_effnetv2s", seed_list=[42], device_idx=1)
    # litmodule_to_trt(CrossViewCFG.output_dir, seed=42)

