from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.datasets import BaseDataset
from src.frameworks.utils.samplers import BaseSampler
from src.models.metrics import BaseMetric


class WrappedDataLoader:
    """Wraps DataLoader to move batches to a specified device."""

    def __init__(self, dataloader: DataLoader, device: str) -> None:
        """
        Args:
            dataloader (DataLoader): DataLoader to wrap.
            device (str): Device to move data to.
        """
        self.dataloader = dataloader
        self.device = device

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.dataloader)

    def __iter__(self):
        """Yield batches moved to device."""
        for inputs, targets in self.dataloader:
            yield inputs.to(self.device), targets.to(self.device)

class BaseClient(ABC):
    """Abstract base class for a federated learning client."""

    def __init__(
        self,
        client_id: str,
        model: Module,
        device: str,
        train_dataset: BaseDataset,
        test_dataset: BaseDataset,
        batch_size: int,
        **dataloader_params: Any,
    ) -> None:
        """
        Initialize client with model, datasets, and dataloaders.

        Args:
            client_id (str): Unique client identifier.
            model (Module): Model instance.
            device (str): Computation device.
            train_dataset (BaseDataset): Training data.
            test_dataset (BaseDataset): Testing data.
            batch_size (int): Batch size.
            **dataloader_params: Extra DataLoader parameters.
        """
        self._id = client_id
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self._setup_train_dataloader(batch_size, **dataloader_params)
        self._setup_test_dataloader(batch_size, **dataloader_params)

    def __len__(self) -> int:
        """Size of training dataset."""
        return len(self.train_dataset)

    @property
    def id(self) -> str:
        """Client ID."""
        return self._id

    def _setup_train_dataloader(self, batch_size: int, **dataloader_params: Any) -> None:
        """Create training dataloader."""
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_params,
        )
        self.train_dataloader = WrappedDataLoader(dataloader, self.device)

    def _setup_test_dataloader(self, batch_size: int, **dataloader_params: Any) -> None:
        """Create testing dataloader."""
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=min(2 * batch_size, len(self.test_dataset)),
            shuffle=False,
            drop_last=False,
            **dataloader_params,
        )

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the local model on the client's training dataset.

        This method must be implemented by concrete client classes and defines
        how local training is performed (e.g., number of epochs, optimizer steps,
        loss computation, gradient updates, or any client-specific behavior).
        Implementations should update the client's model weights accordingly.

        Returns:
            Dict[str, Any]: A dictionary containing training-related outputs.
                Typical entries may include:
                - "num_samples": Number of samples used for training.
                - "model_state_dict": The updated model parameters after training.
                - Additional metadata such as training loss or statistics, 
                  depending on the implementation.
        """
        pass

    @torch.no_grad()
    def evaluate(
        self,
        model_state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Evaluate model and return raw predictions and targets.

        Args:
            model_state_dict (Optional[OrderedDict[str, torch.Tensor]]): Optional model weights.

        Returns:
            Tuple[List[List[int]], List[int]]: Raw predictions and targets.
        """
        if model_state_dict is not None: 
            self.model.load_state_dict(model_state_dict, strict=False) 
        
        self.model.to(self.device)
        self.model.eval()

        self.test_dataset.load()
        preds, targets = [], []

        for x, y in self.test_dataloader: 
            targets.extend(y.tolist())
            preds.extend(self.model(x.to(self.device)).cpu().tolist())

        self.test_dataset.clear()
        self.model.cpu()

        return preds, targets  # This class returns raw preds and targets for external evaluation
    
class BaseServer(ABC): 

    def __init__(
            self,
            clients: List[BaseClient],
            model: Module,
    ):
        self.clients = {client.id: client for client in clients}
        self.model = model
        self.model_state_dict = deepcopy(model.state_dict())

    @abstractmethod 
    def run_round(self, **kwargs: Any) -> Any: 
        pass

    def evaluate(
        self,
        metrics: List[BaseMetric],
        use_local_models: bool = False,
        return_local_evaluations: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all clients using either the global model or each client's local model.

        Args:
            metrics (List[BaseMetric]): Metrics to compute on predictions and targets.
            use_local_models (bool): If True, each client evaluates using its own local model
                instead of the global model.
            return_local_evaluations (bool): Whether to return per-client metric results.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing global metrics under "global"
            and, if requested, per-client metrics under "local".
        """
        all_preds = []
        all_targets = []
        local_results = {}

        for client_id, client in self.clients.items():
            preds, targets = client.evaluate() if use_local_models else client.evaluate(self.model_state_dict)
            all_preds.extend(preds)
            all_targets.extend(targets)

            if return_local_evaluations:
                local_results[client_id] = {
                    metric.id: metric(preds, targets) for metric in metrics
                }

        global_results = {
            metric.id: metric(all_preds, all_targets) for metric in metrics
        }

        results = {"global": global_results}
        if return_local_evaluations:
            results["local"] = local_results

        return results