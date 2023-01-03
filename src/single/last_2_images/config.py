from pathlib import Path
from cfg.general import GeneralCFG


class Last2ImagesCFG:
    output_dir = Path("/workspace", "output", "single", "last_2_images")
    model_name = "resnet50"
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
    Last2ImagesCFG.uploaded_model_dir = Path("aaaaa")
    Last2ImagesCFG.batch_size = 4


if GeneralCFG.debug:
    Last2ImagesCFG.epochs = 2
    Last2ImagesCFG.output_dir = Path("/workspace", "output", "single", "debug_last_2_images")
    Last2ImagesCFG.val_check_interval = GeneralCFG.num_use_data // Last2ImagesCFG.batch_size
