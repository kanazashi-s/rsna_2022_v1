from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from single.two_view_concat.data_module import DataModule


def inspect_dataloader():
    data_module = DataModule(
        seed=42,
        fold=0,
        batch_size=8,
        num_workers=4,
    )
    data_module.setup()

    labels_list = []
    prediction_ids_list = []

    for batch in tqdm(data_module.train_dataloader()):
        inputs, labels = batch
        labels_list.append(labels)

    return labels_list, prediction_ids_list


if __name__ == "__main__":
    labels_list, prediction_ids_list = inspect_dataloader()
    print(labels_list)
    np.unique(torch.cat(labels_list, dim=0).squeeze().numpy(), return_counts=True)
    pd.Series(np.concatenate(prediction_ids_list, axis=0).squeeze()).value_counts()
