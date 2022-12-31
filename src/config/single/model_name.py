from pathlib import Path
from config.general import GeneralCFG


class ModelNameCFG:
    output_dir = Path("/workspace", "output", "single", "model_name")
    model_name = "model_name"
    lr = 2e-5
    backbone_lr_ratio = 0.1
    pool_lr_ratio = 1.0
    fc_lr_ratio = 1.0
    batch_size = 4
    epochs = 4
    max_grad_norm = 1000
    accumulate_grad_batches = 4
    init_weight = "orthogonal"
    val_check_interval = 300
    optimizer_parameters = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    weight_decay = 0.01


if GeneralCFG.debug:
    ModelNameCFG.epochs = 2
    ModelNameCFG.output_dir = Path("/workspace", "output", "single", "debug_model_name")
    ModelNameCFG.val_check_interval = GeneralCFG.num_use_data // 8
