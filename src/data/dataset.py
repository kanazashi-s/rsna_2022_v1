import torch
from torch.utils.data import Dataset, DataLoader
from config.general import GeneralCFG
from config.single.model_name import ModelNameCFG


def prepare_input(inputs):
    return inputs


class TrainDataset(Dataset):
    def __init__(self, input_df):
        pass

    def __len__(self):
        return None

    def __getitem__(self, item):
        inputs = prepare_input(None)
        label = torch.tensor(None)
        return inputs, label


class TestDataset(Dataset):
    def __init__(self, input_df):
        pass

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(None)
        return inputs
