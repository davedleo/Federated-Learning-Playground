from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from src.frameworks.utils import base


class DummyClient(base.BaseClient):
    def train_code(self):
        return "train_code called"

    def train(self):
        return "train called"

    def evaluate(self, model_state_dict=None):
        return [[3, 4, 5]], [2]


def test_wrapped_dataloader_length_and_iteration():
    data = torch.arange(10).unsqueeze(1).float()
    target = torch.zeros(10)  # dummy target tensor
    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")
    wrapped = base.WrappedDataLoader(dataloader, device)

    # test length
    assert len(wrapped) == len(dataloader)

    # test iteration and device transfer
    for batch in wrapped:
        # batch is a tuple with two tensors (input, target)
        assert batch[0].device == device


def test_baseclient_setup_dataloaders():
    device = torch.device("cpu")
    data = torch.arange(4).unsqueeze(1).float()
    target = torch.zeros(4)
    train_dataset = TensorDataset(data, target)
    test_dataset = TensorDataset(data, target)
    model = nn.Linear(1, 1)
    client = DummyClient(client_id="dummy", model=model, device=device, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=2)

    # Check train_dataloader is WrappedDataLoader and test_dataloader is DataLoaderå
    assert isinstance(client.train_dataloader, base.WrappedDataLoader)
    assert isinstance(client.test_dataloader, DataLoader)

    # Check dataloaders length matches expected independently
    import math
    batch_size = 2
    train_batches = math.ceil(len(train_dataset) / batch_size)
    test_batch_size = min(2 * batch_size, len(test_dataset))
    test_batches = math.ceil(len(test_dataset) / test_batch_size)
    assert len(client.train_dataloader) == train_batches
    assert len(client.test_dataloader) == test_batches


def test_train_and_evaluate_methods():
    device = torch.device("cpu")
    data = torch.arange(4).unsqueeze(1).float()
    target = torch.zeros(4)
    train_dataset = TensorDataset(data, target)
    test_dataset = TensorDataset(data, target)
    model = nn.Linear(1, 1)
    client = DummyClient(client_id="dummy", model=model, device=device, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=2)
    assert client.train() == "train called"
    assert client.evaluate() == ([[3, 4, 5]], [2])


class DummyServer(base.BaseServer):
    def run_round(self, client_criterion, client_optimizer, num_epochs, **kwargs):
        return {"status": "ok"}

def test_baseserver_evaluate_global_and_local():
    device = torch.device("cpu")
    data = torch.arange(4).unsqueeze(1).float()
    target = torch.zeros(4)

    train_dataset = TensorDataset(data, target)
    test_dataset = TensorDataset(data, target)

    class DS(base.BaseDataset):
        def __init__(self, t, y):
            self.t = t
            self.y = y
        def load(self): pass
        def clear(self): pass
        def load_data(self): pass
        def __len__(self): return len(self.t)
        def __getitem__(self, idx): return self.t[idx], self.y[idx]

    # wrap into BaseDataset-compatible
    train_ds = DS(data, target)
    test_ds = DS(data, target)

    model = nn.Linear(1, 1)

    client1 = DummyClient(
        client_id="c1",
        model=deepcopy(model),
        device=device,
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=2
    )
    client2 = DummyClient(
        client_id="c2",
        model=deepcopy(model),
        device=device,
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=2
    )

    server = DummyServer([client1, client2], model)

    class DummyMetric(base.BaseMetric):
        def __init__(self): super().__init__("dummy")
        def forward(self, preds, targets): return len(preds) + len(targets)

    metrics = [DummyMetric()]

    results = server.evaluate(metrics, use_local_models=False, return_local_evaluations=True)

    assert "global" in results
    assert "local" in results
    assert "c1" in results["local"]
    assert "c2" in results["local"]
    assert results["global"]["dummy"] == 4
