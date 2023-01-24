import os
from pathlib import Path
from cfg.general import GeneralCFG


class TwoViewConcatCFG:
    output_dir = Path("/workspace", "output", "single", "two_view_concat")
    upload_name = "two-view-concat-baseline-20220118"
    model_name = "efficientnet_b0"
    drop_rate = 0.3
    drop_path_rate = 0.2
    lr = 3e-4
    backbone_lr_ratio = 1.0
    pool_lr_ratio = 1.0
    fc_lr_ratio = 1.0
    batch_size = 8
    epochs = 4
    max_grad_norm = 10
    accumulate_grad_batches = 16
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
    sampler = "ImbalancedDatasetSampler"
    num_samples_per_epoch = None

    monitor_metric = "best_pfbeta"
    monitor_mode = "max"
    uploaded_model_dir = output_dir


if GeneralCFG.is_kaggle:
    TwoViewConcatCFG.uploaded_model_dir = Path("/kaggle", "input", TwoViewConcatCFG.upload_name)
    TwoViewConcatCFG.batch_size = 4


if GeneralCFG.debug:
    TwoViewConcatCFG.epochs = 2
    TwoViewConcatCFG.output_dir = Path("/workspace", "output", "single", "debug_two_view_concat")
