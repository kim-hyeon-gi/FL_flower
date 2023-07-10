from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from torch.utils.tensorboard import SummaryWriter

from model.utils import load_model, model_pretraining, test
from utils import get_parameters, set_parameters


class FedCustom(fl.server.strategy.Strategy):
    def __init__(self, config, test_loader, writer) -> None:
        super().__init__()
        self.test_loader = test_loader
        self.config = config
        self.fraction_fit = config.fraction_fit
        self.fraction_evaluate = config.fraction_evaluate
        self.min_fit_clients = config.min_fit_clients
        self.min_evaluate_clients = config.min_evaluate_clients
        self.min_available_clients = config.min_available_clients
        self.pt_path = config.model_pt_path
        if self.pt_path == "None":
            self.pt_path = model_pretraining(config)
        self.model = torch.load(self.pt_path).to(config.server_device)
        self.writer = writer

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_parameters(self.model)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {
            "lr": 0.001,
            "server_round": server_round,
            "local_epochs": 3,
            "device": self.config.device,
        }
        higher_lr_config = {
            "lr": 0.003,
            "server_round": server_round,
            "local_epochs": 3,
            "device": self.config.device,
        }
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        """Evaluate global model parameters using an evaluation function."""
        test_loader = self.test_loader
        parameters = parameters_to_ndarrays(parameters)
        set_parameters(
            self.model, parameters
        )  # Update model with the latest parameters
        loss, accuracy = test(self.model, test_loader, self.config.server_device)
        print(
            f"Server-side evaluation round {server_round} loss {loss} / accuracy {accuracy}"
        )
        self.writer.add_scalar("acc/round", accuracy, server_round)
        self.writer.add_scalar("loss/round", loss, server_round)
        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
