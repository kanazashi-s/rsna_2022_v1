import os
from pathlib import Path
from cfg.general import GeneralCFG


class MeanAggCFG:
    output_dir = Path("/workspace", "output", "single", "mean_agg")
    upload_name = "mean-agg-baseline-20220118"
    model_name = "efficientnet_b0"
    drop_rate = 0.3
    drop_path_rate = 0.2
    lr = 3e-4
    backbone_lr_ratio = 1.0
    pool_lr_ratio = 1.0
    fc_lr_ratio = 1.0
    batch_size = 8
    epochs = 20
    max_grad_norm = 10
    accumulate_grad_batches = 8
    init_weight = "orthogonal"
    optimizer_parameters = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    weight_decay = 0.01
    warmup_epochs = 1

    loss_function = "SigmoidFocalLoss"
    pos_weight = [1.0]  # only used when loss_function == "BCEWithLogitsLoss"
    focal_loss_alpha = 1.0  # only used when loss_function == "SigmoidFocalLoss"
    focal_loss_gamma = 2.0  # only used when loss_function == "SigmoidFocalLoss"
    sampler = "AtLeastOnePositiveSampler"  # None, "ImbalancedDatasetSampler" or "AtLeastOnePositiveSampler"
    num_samples_per_epoch = None

    monitor_metric = "auc_pr"
    monitor_mode = "max"
    uploaded_model_dir = output_dir
    val_check_per_epoch = 2


if GeneralCFG.is_kaggle:
    MeanAggCFG.uploaded_model_dir = Path("/kaggle", "input", MeanAggCFG.upload_name)
    MeanAggCFG.batch_size = 4


if GeneralCFG.debug:
    MeanAggCFG.epochs = 2
    MeanAggCFG.output_dir = Path("/workspace", "output", "single", "debug_mean_agg")
