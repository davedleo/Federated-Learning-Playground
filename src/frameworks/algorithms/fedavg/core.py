from copy import deepcopy

from torch import Tensor
from torch.nn import Module 
from torch.optim import Optimizer 

from typing import Any, Dict, List, Union

from src.frameworks.utils.base import BaseClient, BaseServer
from src.frameworks.utils.samplers import UniformSampler


class Client(BaseClient):
    """Client participating in a federated learning setup, responsible for performing local training on private data and producing model updates for aggregation."""

    def train(
            self,
            num_epochs: int,
            criterion: Module,
            optimizer_cls: Optimizer,
            model_state_dict: Dict[str, Any] = None,
            **optimizer_params: Any
    ) -> Dict[str, Any]:
        """
        Perform local training for a specified number of epochs and return updated model parameters.

        This method optionally loads a provided global model state, trains locally using the client's
        private dataset, and returns model updates alongside metadata such as the number of samples used.

        Args:
            num_epochs (int): Number of full passes over the local dataset.
            criterion (Module): Loss function used to compute training loss.
            optimizer_cls (Optimizer): Optimizer class constructed for local updates.
            model_state_dict (Dict[str, Any], optional): State dictionary containing parameters of the current
                global model. If provided, the client loads these weights before beginning training.
            **optimizer_params (Any): Additional keyword arguments forwarded to the optimizer constructor.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "model": The updated local model state dictionary after training.
                - "n_samples": Number of samples used during local training.
        """
        self.train_dataset.load()

        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model.to(self.device)
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        criterion.to(self.device)

        for _ in range(num_epochs): 
            for inputs, targets in self.train_dataloader: 
                optimizer.zero_grad()
                outputs: Tensor = self.model(inputs)
                loss: Tensor = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        self.train_dataset.clear()
        self.model.cpu()
        criterion.cpu()

        del criterion, optimizer

        return {"model": deepcopy(self.model.state_dict()), "n_samples": len(self.train_dataset)}
    
class Server(BaseServer): 
    """Server coordinating federated learning rounds, including client selection, distribution of the global model, collection of updates, and execution of FedAvg aggregation."""

    def __init__(
            self, 
            clients: List[Client],
            model: Module
    ): 
        """
        Initialize the federated learning server.

        Args:
            clients (List[Client]): All available clients participating in FL.
            model (Module): The global model whose parameters will be aggregated across rounds.

        Notes:
            The server also initializes a uniform sampler to select participating clients each round.
        """
        super().__init__(clients, model)
        self.sampler = UniformSampler()

    def _aggregate(self, updates) -> None:
        """
        Aggregate client updates using weighted FedAvg and update the server's global model.
        """
        aggregated_state = deepcopy(self.model_state_dict)
        total_samples = sum(update["n_samples"] for update in updates)

        for key in aggregated_state.keys():
            # Skip BatchNorm statistics and buffers
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                continue

            weighted_sum = None
            for update in updates:
                client_state = update["model"][key]
                weight = update["n_samples"] / total_samples
                if weighted_sum is None:
                    weighted_sum = client_state * weight
                else:
                    weighted_sum += client_state * weight

            aggregated_state[key] = weighted_sum

        self.model.load_state_dict(aggregated_state, strict=False)
        self.model_state_dict = deepcopy(aggregated_state)

    def run_round(
            self,
            num_epochs: int,
            participation_rate: Union[int, float],
            criterion: Module, 
            optimizer_cls: Optimizer,
            **optimizer_params: Any
    ): 
        """
        Execute a full federated learning round.

        Args:
            num_epochs (int): Number of local training epochs for each selected client.
            participation_rate (int or float): Number or proportion of clients to sample.
            criterion (Module): Loss function used during client training.
            optimizer_cls (Optimizer): Optimizer class instantiated on each client.
            **optimizer_params (Any): Additional parameters passed to the optimizer constructor.

        Returns:
            None: The server updates its internal global model state in-place.

        Notes:
            This method performs:
                1. Sampling a subset of clients.
                2. Sending the current global model to selected clients.
                3. Running local training on each client.
                4. Collecting updated model parameters.
                5. Aggregating updates using weighted FedAvg (excluding BatchNorm statistics).
        """
        # Select clients 
        n_clients = participation_rate if isinstance(participation_rate, int) else int(participation_rate * len(self.clients))
        all_client_ids = list(self.clients.keys())
        client_ids = self.sampler(all_client_ids, n_clients)

        # Train clients 
        updates = []
        for client_id in client_ids: 
            update = self.clients[client_id].train(
                model_state_dict=deepcopy(self.model_state_dict),
                num_epochs=num_epochs,
                criterion=criterion,
                optimizer_cls=optimizer_cls,
                **optimizer_params
            )

            updates.append(update)

        self._aggregate(updates)
