import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class OnePositiveSampler(BatchSampler):
    def __init__(self, indices, labels, batch_size, sampler=None):
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.sampler = sampler
        self.remaining_indices = np.arange(len(self.indices))
        self.positive_indices = self.indices[self.labels == 1]

    def __iter__(self):
        while self.remaining_indices.size > 0:
            if len(np.intersect1d(self.remaining_indices, self.positive_indices)) == 0:
                self.remaining_indices = np.concatenate((self.remaining_indices, self.positive_indices))
                print("Running out of positive samples. Added all positive samples to remaining indices.")

            if len(np.setdiff1d(self.remaining_indices, self.positive_indices)) < self.batch_size - 1:
                subset = np.setdiff1d(self.remaining_indices, self.positive_indices)
                self.remaining_indices = np.arange(len(self.indices))
                print(f"Num of neg samples of this batch is {len(subset)}. \nRemaining indices has been reset!")
            else:
                rng = np.random.default_rng()
                # Get a random subset of remaining indices that doesn't include any positive indices
                subset = rng.choice(
                    np.setdiff1d(self.remaining_indices, self.positive_indices),
                    self.batch_size - 1,
                    replace=False
                )
            # Get a random positive index
            positive_index = rng.choice(
                np.intersect1d(self.remaining_indices, self.positive_indices),
                replace=False
            )

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


class StratifiedOnePositiveSampler(BatchSampler):
    def __init__(self, indices, labels, batch_size, groups):
        assert len(indices) == len(labels) == len(groups)
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.groups = groups
        self.remaining_indices = np.arange(len(self.indices))
        self.positive_indices = self.indices[self.labels == 1]

        self.grouped_indices = {}
        for group in np.unique(self.groups):
            self.grouped_indices[group] = np.where(self.groups == group)[0]

    def __iter__(self):
        while self.remaining_indices.size > 0:
            if len(self.positive_indices) == 0:
                raise StopIteration("No positive samples.")

            if len(self.remaining_indices) < self.batch_size:
                self.remaining_indices = np.arange(len(self.indices))
                print("remaining indices has been reset!")
                continue

            else:
                # Get unique groups from the remaining indices
                unique_groups = np.unique(self.groups[self.remaining_indices])
                # ensure that there is at least one positive index in the group
                # and that there is at least one negative index in the group
                for group in unique_groups:
                    group_indices = np.where(self.groups == group)[0]
                    group_indices = np.intersect1d(group_indices, self.remaining_indices)
                    group_positive_indices = np.intersect1d(group_indices, self.positive_indices)
                    group_negative_indices = np.setdiff1d(group_indices, self.positive_indices)
                    if len(group_positive_indices) == 0 or len(group_negative_indices) == 0:
                        unique_groups = np.setdiff1d(unique_groups, group)
                if len(unique_groups) == 0:
                    self.remaining_indices = np.arange(len(self.indices))
                    print("remaining indices has been reset!")
                    continue

                # Select a random group to sample from
                group = np.random.choice(unique_groups)
                group_indices = np.where(self.groups == group)[0]
                group_indices = np.intersect1d(group_indices, self.remaining_indices)

                # Get a random subset of remaining indices that doesn't include any positive indices
                # and is from the selected group
                group_negative_indices = np.setdiff1d(group_indices, self.positive_indices)
                if len(group_negative_indices) < self.batch_size - 1:
                    subset = group_negative_indices
                else:
                    subset = np.random.choice(group_negative_indices, self.batch_size - 1, replace=False)

                # Get a random positive index from the selected group
                group_positive_indices = np.intersect1d(group_indices, self.positive_indices)
                positive_index = np.random.choice(group_positive_indices)

                # Concatenate the positive index to the subset
                subset = np.concatenate((subset, [positive_index]))

            # ensure that the groups of the subset are all the same
            assert len(np.unique(self.groups[subset])) == 1

            self.remaining_indices = np.setdiff1d(self.remaining_indices, subset)
            yield subset

    def __len__(self):
        length = len(self.indices) // self.batch_size
        if len(self.positive_indices) < length:
            return len(self.positive_indices)
        else:
            return length


class StratifiedSampler(BatchSampler):
    def __init__(self, indices, groups, batch_size):
        self.indices = indices
        self.batch_size = batch_size
        self.groups = groups
        self.remaining_indices = np.arange(len(self.indices))
        self.grouped_indices = {}
        for group in np.unique(self.groups):
            self.grouped_indices[group] = np.where(self.groups == group)[0]

    def __iter__(self):
        while self.remaining_indices.size > 0:
            if len(self.remaining_indices) < self.batch_size:
                self.remaining_indices = np.arange(len(self.indices))
                print("remaining indices has been reset!")
                # raise StopIteration("run out samples.")

            else:
                # Get unique groups from the remaining indices
                unique_groups = np.unique(self.groups[self.remaining_indices])

                # Select a random group to sample from
                group = np.random.choice(unique_groups)
                group_indices = np.where(self.groups == group)[0]
                group_indices = np.intersect1d(group_indices, self.remaining_indices)

                # Get a random subset of indices that is from the selected group
                if len(group_indices) < self.batch_size:
                    subset = group_indices
                else:
                    subset = np.random.choice(group_indices, self.batch_size, replace=False)

            yield self.indices[subset]
            self.remaining_indices = np.setdiff1d(self.remaining_indices, subset)

    def __len__(self):
        return len(self.indices) // self.batch_size


if __name__ == "__main__":
    # Test StratifiedOnePositiveSampler
    indices = np.arange(300)
    rng = np.random.RandomState(0)
    labels = rng.choice([0, 1], 300, p=[0.95, 0.05])
    groups = rng.choice([100, 200, 300], 300, p=[0.6, 0.3, 0.1])
    sampler = StratifiedOnePositiveSampler(indices, labels, 8, groups)
    # for batch in sampler:
    #     # ensure that the groups of the batch are all the same
    #     assert len(np.unique(groups[batch])) == 1
    #     # ensure that there is one positive index in the batch
    #     assert np.sum(labels[batch]) == 1
    #     # ensure that there is at least one negative index in the batch
    #     assert np.sum(labels[batch]) < len(batch)

    # Test StratifiedSampler
    indices = np.arange(300)
    rng = np.random.RandomState(0)
    labels = rng.choice([0, 1], 300, p=[0.95, 0.05])
    groups = rng.choice([100, 200, 300], 300, p=[0.6, 0.3, 0.1])
    sampler = StratifiedSampler(indices, groups, 8)
    for batch in sampler:
        # ensure that the groups of the batch are all the same
        assert len(np.unique(groups[batch])) == 1
        print(batch)

