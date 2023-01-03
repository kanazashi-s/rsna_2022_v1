from pathlib import Path
from cfg.general import GeneralCFG


class MeanImagesCFG:
    output_dir = Path("/workspace", "output", "single", "mean_images")
    model_name = "efficientnetv2_rw_m"
    lr = 3e-4
    backbone_lr_ratio = 1.0
    pool_lr_ratio = 1.0
    fc_lr_ratio = 1.0
    batch_size = 16
    epochs = 4
    max_grad_norm = 10
    accumulate_grad_batches = 8
    init_weight = "orthogonal"
    optimizer_parameters = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    weight_decay = 0.01

    loss_function = "BCEWithLogitsLoss"
    pos_weight = [1.0]  # only used when loss_function == "BCEWithLogitsLoss"
    focal_loss_alpha = 1.0  # only used when loss_function == "SigmoidFocalLoss"
    focal_loss_gamma = 2.0  # only used when loss_function == "SigmoidFocalLoss"
    monitor_metric = "score"
    monitor_mode = "max"
    uploaded_model_dir = None


if GeneralCFG.is_kaggle:
    MeanImagesCFG.uploaded_model_dir = Path("aaaaa")
    MeanImagesCFG.batch_size = 4


if GeneralCFG.debug:
    MeanImagesCFG.epochs = 2
    MeanImagesCFG.output_dir = Path("/workspace", "output", "single", "debug_mean_images")
    MeanImagesCFG.val_check_interval = GeneralCFG.num_use_data // MeanImagesCFG.batch_size
