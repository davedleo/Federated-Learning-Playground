import numpy as np
from src.data.distribution import Dirichlet

np.random.seed(42)

def generate_data(num_samples=100, num_classes=3):
    """Generate synthetic dataset with given number of classes and samples."""
    data = []
    for i in range(num_samples):
        label = i % num_classes
        data.append((f"x{i}", label))
    return data

def test_sampler_output_format_and_nonempty():
    train = generate_data(60, 3)
    test = generate_data(30, 3)

    sampler = Dirichlet(train, test, alpha=0.5)
    result = sampler.sample(n_splits=5)

    # Check output format
    assert isinstance(result, list)
    assert len(result) == 5
    for client in result:
        assert "train" in client and "test" in client
        assert isinstance(client["train"], list)
        assert isinstance(client["test"], list)
        # Check no client is empty
        assert len(client["train"]) > 0
        assert len(client["test"]) > 0

def test_dirichlet_alpha_effect():
    train = generate_data(90, 3)
    test = generate_data(45, 3)

    sampler_low = Dirichlet(train, test, alpha=1.0)
    sampler_high = Dirichlet(train, test, alpha=1000.0)

    low = sampler_low.sample(3)
    high = sampler_high.sample(3)

    low_sizes = [len(c["train"]) for c in low]
    high_sizes = [len(c["train"]) for c in high]

    # Low alpha should give more skewed sizes
    assert np.std(low_sizes) > np.std(high_sizes)

def test_large_number_of_splits_no_empty():
    train = generate_data(200, 5)
    test = generate_data(100, 5)

    sampler = Dirichlet(train, test, alpha=0.8)
    result = sampler.sample(n_splits=20)

    for client in result:
        assert len(client["train"]) > 0
        assert len(client["test"]) > 0