import numpy as np
import pytest
from utils.model_utils.torch_samplers import StratifiedOnePositiveSampler

@pytest.fixture
def data():
    indices = np.arange(20)
    labels = np.concatenate((np.zeros(10), np.ones(10)))
    groups = np.concatenate((np.zeros(5), np.ones(5), np.ones(5) * 2, np.ones(5) * 3))
    return indices, labels, groups

def test_sampler_batch_size(data):
    indices, labels, groups = data
    batch_size = 5
    sampler = StratifiedOnePositiveSampler(indices, labels, batch_size, groups)
    batches = [batch for batch in sampler]
    assert len(batches) == 4  # 4 because 20 samples / 5 batch_size = 4
    for batch in batches:
        assert len(batch) == batch_size

def test_sampler_stratified(data):
    indices, labels, groups = data
    batch_size = 5
    sampler = StratifiedOnePositiveSampler(indices, labels, batch_size, groups)
    batches = [batch for batch in sampler]
    for batch in batches:
        assert np.sum(labels[batch]) >= 1  # at least one positive sample in each batch

def test_sampler_reset(data):
    indices, labels, groups = data
    batch_size = 5
    sampler = StratifiedOnePositiveSampler(indices, labels, batch_size, groups)
    batches = [batch for batch in sampler]
    for batch in batches:
        pass
    batches2 = [batch for batch in sampler]
    assert len(batches) == len(batches2)  # remaining_indices should be reset after one iteration
    for batch, batch2 in zip(batches, batches2):
        assert np.all(batch == batch2)  # batches should be the same after resetting the remaining_indices
