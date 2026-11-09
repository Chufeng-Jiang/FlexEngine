import torch

from flexengine.sampling.sampler import sample


def test_greedy_and_rng_reproducibility():
    x = torch.tensor([-10.0, 2.0, 9.0, 1.0])
    assert sample(x, 0.0, 0) == 2
    a, b = torch.Generator().manual_seed(55), torch.Generator().manual_seed(55)
    assert [sample(x, 1.0, 2, a) for _ in range(25)] == [sample(x, 1.0, 2, b) for _ in range(25)]
