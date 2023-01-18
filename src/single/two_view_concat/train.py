from pathlib import Path
import shutil
import gc
from collections import defaultdict
import numpy as np
import torch
import pytorch_lightning as pl
from cfg.general import GeneralCFG
from single.two_view_concat.config import TwoViewConcatCFG
from single.two_view_concat.data_module import DataModule
from single.two_view_concat.model.lit_module import LitModel
from single.two_view_concat.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def train(run_name: str, seed_list=None, device_idx=0):

    metrics_seed_dict = defaultdict(list)
    if seed_list is None:
        seed_list = GeneralCFG.seeds

    for seed in seed_list:
        output_dir = TwoViewConcatCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = DataModule(
                seed=seed,
                fold=fold,
                batch_size=TwoViewConcatCFG.batch_size,
                num_workers=GeneralCFG.num_workers,
            )
            data_module.setup()

            model = LitModel()

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + f"two_view_concat_{TwoViewConcatCFG.model_name}",
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            val_check_interval = len(data_module.train_dataloader()) // 5

            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor=TwoViewConcatCFG.monitor_metric,
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode=TwoViewConcatCFG.monitor_mode,
                # every_n_train_steps=val_check_interval,
                verbose=True,
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            callbacks = [loss_callback, lr_monitor]

            trainer = pl.Trainer(
                devices=[device_idx],
                accelerator="gpu",
                max_epochs=TwoViewConcatCFG.epochs,
                precision="bf16",
                amp_backend='native',
                gradient_clip_val=TwoViewConcatCFG.max_grad_norm,
                accumulate_grad_batches=TwoViewConcatCFG.accumulate_grad_batches,
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
            model_name=f"{TwoViewConcatCFG.upload_name}-seed{seed}",
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
        model_name=TwoViewConcatCFG.upload_name,
        model_path=TwoViewConcatCFG.output_dir,
    )

    return metrics_seed_mean_dict


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": TwoViewConcatCFG.model_name,
        "lr": TwoViewConcatCFG.lr,
        "batch_size": TwoViewConcatCFG.batch_size,
        "epochs": TwoViewConcatCFG.epochs,
        "max_grad_norm": TwoViewConcatCFG.max_grad_norm,
        "accumulate_grad_batches": TwoViewConcatCFG.accumulate_grad_batches,
        "loss_function": TwoViewConcatCFG.loss_function,
        "pos_weight": TwoViewConcatCFG.pos_weight,
        "focal_loss_alpha": TwoViewConcatCFG.focal_loss_alpha,
        "focal_loss_gamma": TwoViewConcatCFG.focal_loss_gamma,
        "monitor_metric": TwoViewConcatCFG.monitor_metric,
        "monitor_mode": TwoViewConcatCFG.monitor_mode,
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
    # TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "baseline_512")
    # oof_pfbeta_seed_mean = train(f"two_view_concat_baseline", seed_list=[42], device_idx=0)

    # GeneralCFG.num_workers = 4
    # TwoViewConcatCFG.batch_size = 16
    # TwoViewConcatCFG.accumulate_grad_batches = 8
    # TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "roi_extracted_512")
    # GeneralCFG.image_size = 512
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "roi_extracted_512"
    # train(f"two_view_concat_roi_extracted_512", seed_list=[42, 43, 44], device_idx=0)

    GeneralCFG.num_workers = 4
    TwoViewConcatCFG.batch_size = 16
    TwoViewConcatCFG.accumulate_grad_batches = 8
    TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "baseline_512")
    GeneralCFG.image_size = 512
    GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_512"
    train(f"two_view_concat_baseline_512", seed_list=[42, 43, 44], device_idx=1)

    # TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "baseline_1024")
    # GeneralCFG.image_size = 1024
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_1024"
    # GeneralCFG.num_workers = 4
    # train(f"two_view_concat_baseline_1024", seed_list=[43, 44], device_idx=1)

    # epoch = 1
    # TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "epoch_1")
    # GeneralCFG.image_size = 512
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_512"
    # TwoViewConcatCFG.epochs = 1
    # train(f"two_view_concat_epoch_1", seed_list=[42, 43, 44], device_idx=0)
    #
    # # Under sampling
    # TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "two_view_concat", "under_sampling")
    # GeneralCFG.image_size = 512
    # GeneralCFG.train_image_dir = GeneralCFG.png_data_dir / "theo_512"
    # TwoViewConcatCFG.epochs = 4
    # TwoViewConcatCFG.num_samples_per_epoch = 1000
    # train(f"two_view_concat_under_sampling", seed_list=[42, 43, 44], device_idx=0)





