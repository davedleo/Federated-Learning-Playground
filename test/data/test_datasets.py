import pytest
import torch
from pathlib import Path
from PIL import Image
from unittest import mock

from src.data.datasets import BaseDataset, TorchDataset, RGBDataset, BWDataset


class DummyDataset(BaseDataset):
    def load_data(self, path):
        # Simply return a tensor containing the length of the filename for testing
        return torch.tensor([len(str(path))], dtype=torch.float)


def test_abstract_dataset_length():
    dataset = DummyDataset([(torch.tensor([1.0]), 0), (torch.tensor([2.0]), 1)], use_cache=False)
    assert len(dataset) == 2


def test_no_cache_loading():
    dataset = DummyDataset([(torch.tensor([5.0]), 1)], use_cache=False)
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([5.0]))
    assert y.item() == 1


def test_with_cache_loading():
    dataset = DummyDataset([(torch.tensor([3.0]), 2)], use_cache=True)
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([3.0]))
    assert y.item() == 2


def test_cache_clear():
    dataset = DummyDataset([(torch.tensor([7.0]), 1)], use_cache=True)
    assert len(dataset.cache) == 1
    dataset.clear()
    assert len(dataset.cache) == 0


def test_transform_called():
    mock_transform = mock.Mock(side_effect=lambda x: x + 1)
    dataset = DummyDataset([(torch.tensor([1.0]), 0)], transforms=[mock_transform], use_cache=False)
    x, _ = dataset[0]
    mock_transform.assert_called_once()
    assert torch.equal(x, torch.tensor([2.0]))


def test_tensor_dataset_valid_extension(tmp_path):
    t = torch.tensor([1.0, 2.0])
    file_path = tmp_path / "test.pt"
    torch.save(t, file_path)

    dataset = TorchDataset([(file_path, 0)], use_cache=False)
    x, _ = dataset[0]
    assert torch.equal(x, t)


def test_tensor_dataset_invalid_extension(tmp_path):
    file_path = tmp_path / "invalid.txt"
    file_path.write_text("bad")

    dataset = TorchDataset([(file_path, 0)], use_cache=False)
    with pytest.raises(ValueError):
        _ = dataset[0]


def create_dummy_image(path: Path, mode: str):
    """Create and save a dummy 10x10 image in the given mode."""
    img = Image.new(mode, (10, 10), color=128)
    img.save(path)


def test_rgb_dataset_loading(tmp_path):
    file_path = tmp_path / "rgb.png"
    create_dummy_image(file_path, "RGB")

    dataset = RGBDataset([(file_path, 0)], use_cache=False)
    x, y = dataset[0]

    assert x.shape == (3, 10, 10)
    assert y.item() == 0
    assert x.min() >= 0 and x.max() <= 1


def test_bw_dataset_loading(tmp_path):
    file_path = tmp_path / "bw.png"
    create_dummy_image(file_path, "L")

    dataset = BWDataset([(file_path, 0)], use_cache=False)
    x, y = dataset[0]

    assert x.shape == (1, 10, 10)
    assert y.item() == 0
    assert x.min() >= 0 and x.max() <= 1


def test_cache_preload():
    ds = DummyDataset([(torch.tensor([2.0]), 1)], use_cache=True)
    assert len(ds.cache) == 1
    assert torch.equal(ds.cache[0][0], torch.tensor([2.0]))


def test_cache_vs_no_cache():
    tensor = torch.tensor([4.0])
    ds_cache = DummyDataset([(tensor, 1)], use_cache=True)
    ds_nocache = DummyDataset([(tensor, 1)], use_cache=False)

    x1, _ = ds_cache[0]
    x2, _ = ds_nocache[0]

    assert torch.equal(x1, x2)
