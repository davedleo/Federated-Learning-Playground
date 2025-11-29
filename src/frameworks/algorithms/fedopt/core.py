from copy import deepcopy

from torch import no_grad
from torch.nn import Module 
from torch.optim import Optimizer 

from typing import Any, Dict, List, Union

from src.frameworks.algorithms.fedavg import Client
from src.frameworks.utils.base import BaseServer
from src.frameworks.utils.samplers import UniformSampler

    
class Server(BaseServer):
    """Server coordinating federated learning rounds with FedOpt aggregation.
    
    The server maintains a global model and a server-side optimizer. Client updates are 
    converted into pseudo-gradients, which are then used to update the global model
    using the specified optimizer (e.g., FedAdam, FedYogi).
    """

    def __init__(
        self, 
        clients: List[Client],
        model: Module,
        optimizer_cls: Optimizer,
        **optimizer_params: Any
    ): 
        """
        Initialize the FedOpt server.

        Args:
            clients (List[Client]): All available clients participating in FL.
            model (Module): The global model whose parameters will be aggregated.
            optimizer_cls (Optimizer): Optimizer class used on the server for global updates.
            **optimizer_params (Any): Parameters passed to the server optimizer.

        Notes:
            A uniform sampler is used to select clients each round.
            The server optimizer is initialized with the global model parameters.
        """
        super().__init__(clients, model)
        self.sampler = UniformSampler()
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.model_state_dict = deepcopy(self.model.state_dict())

    def _aggregate(self, updates: List[Dict[str, Any]]) -> None:
        """
        Aggregate client updates as FedOpt pseudo-gradients and perform a server optimizer step.

        Args:
            updates (List[Dict[str, Any]]): List of client updates, each containing "model" and "n_samples".
        """
        total_samples = sum(update["n_samples"] for update in updates)

        # Compute FedOpt pseudo-gradients
        grads = {}
        for key in self.model_state_dict.keys():
            # Skip BatchNorm statistics and buffers
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                continue

            grad = None
            for update in updates:
                client_param = update["model"][key]
                delta = self.model_state_dict[key] - client_param
                weight = update["n_samples"] / total_samples
                if grad is None:
                    grad = delta * weight
                else:
                    grad += delta * weight

            grads[key] = grad

        # Apply pseudo-gradients to server model
        with no_grad():
            for name, param in self.model.named_parameters():
                if name in grads:
                    param.grad = grads[name]
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model_state_dict = deepcopy(self.model.state_dict())

    def run_round(
        self,
        num_epochs: int,
        participation_rate: Union[int, float],
        criterion: Module, 
        **client_optimizer_params: Any
    ): 
        """
        Execute a full FedOpt federated learning round.

        Args:
            num_epochs (int): Number of local epochs on each selected client.
            participation_rate (int or float): Number or fraction of clients to sample.
            criterion (Module): Loss function used for client training.
            **client_optimizer_params (Any): Additional parameters passed to client optimizers.

        Notes:
            This method:
                1. Samples a subset of clients.
                2. Sends the current global model to selected clients.
                3. Runs local training on each client.
                4. Aggregates updates using server-side optimizer (FedOpt).
        """
        n_clients = participation_rate if isinstance(participation_rate, int) else int(participation_rate * len(self.clients))
        all_client_ids = list(self.clients.keys())
        client_ids = self.sampler(all_client_ids, n_clients)

        updates = []
        for client_id in client_ids: 
            update = self.clients[client_id].train(
                model_state_dict=deepcopy(self.model_state_dict),
                num_epochs=num_epochs,
                criterion=criterion,
                **client_optimizer_params
            )
            updates.append(update)

        self._aggregate(updates)
