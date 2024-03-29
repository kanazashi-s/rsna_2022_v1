import os
from pathlib import Path
from cfg.general import GeneralCFG


class MeanAggCFG:
    output_dir = Path("/workspace", "output", "single", "mean_agg")
    upload_name = "mean-agg-ker-optim-swa-20220213"  # trt変換用Notebookは、この名前 + -trt で作成する
    trt_model_dir = output_dir
    model_name = "efficientnetv2_rw_s"
    drop_rate = 0.2
    drop_path_rate = 0.2
    lr = 1.5e-4
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
    weight_decay = 1e-4
    warmup_epochs = 1

    loss_function = "SigmoidFocalLoss"
    pos_weight = [1.0]  # only used when loss_function == "BCEWithLogitsLoss"
    focal_loss_alpha = 1.0  # only used when loss_function == "SigmoidFocalLoss"
    focal_loss_gamma = 2.0  # only used when loss_function == "SigmoidFocalLoss"
    sampler = "OnePositiveSampler"  # None, "ImbalancedDatasetSampler" or "OnePositiveSampler"
    num_samples_per_epoch = None

    monitor_metric = "auc_pr"
    monitor_mode = "max"
    uploaded_model_dir = output_dir
    val_check_per_epoch = 2
    use_tta = True


if GeneralCFG.is_kaggle:
    MeanAggCFG.uploaded_model_dir = Path("/kaggle", "input", MeanAggCFG.upload_name)
    MeanAggCFG.trt_model_dir = Path("/kaggle", "input", MeanAggCFG.upload_name + "-trt")
    MeanAggCFG.batch_size = 4


if GeneralCFG.debug:
    MeanAggCFG.epochs = 2
    MeanAggCFG.output_dir = Path("/workspace", "output", "single", "debug_mean_agg")
