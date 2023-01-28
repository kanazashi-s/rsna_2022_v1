import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class AtLeastOnePositiveSampler(BatchSampler):
    def __init__(self, indices, labels, batch_size):
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.remaining_indices = np.arange(len(self.indices))
        self.positive_indices = self.indices[self.labels == 1]

    def __iter__(self):
        while self.remaining_indices.size > 0:
            if len(self.positive_indices) == 0:
                raise StopIteration("No positive samples.")

            if len(self.remaining_indices) < self.batch_size:
                subset = self.remaining_indices
                self.remaining_indices = np.arange(len(self.indices))
                print("remaining indices has been reset!")
            else:
                # Get a random subset of remaining indices that doesn't include any positive indices
                subset = np.random.choice(np.setdiff1d(self.remaining_indices, self.positive_indices),
                                          self.batch_size - 1, replace=False)
                positive_index = np.random.choice(self.positive_indices)
                # Concatenate the positive index to the subset
                subset = np.concatenate((subset, [positive_index]))
            yield self.indices[subset]
            self.remaining_indices = np.setdiff1d(self.remaining_indices, subset)

    def __len__(self):
        length = len(self.indices) // self.batch_size
        if len(self.positive_indices) < length:
            return len(self.positive_indices)
        else:
            return length


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
