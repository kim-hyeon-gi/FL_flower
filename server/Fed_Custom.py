import math
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import scipy
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
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from scipy.sparse import csr_array
from torch.utils.tensorboard import SummaryWriter

from communication.base_utils import (
    ndarrays_to_sparse_parameters,
    sparse_parameters_to_ndarrays,
)
from communication.low_rank_utils import low_rank_sparse_parameters_to_ndarrays
from communication.prob_quantization_utils import (
    prob_quantization_sparse_parameters_to_ndarrays,
)
from communication.sparse_utils import (
    efficient_communication_sparse_parameters_to_ndarrays,
)
from model.utils import load_model, model_pretraining, test
from quantization.eden import eden_builder
from utils import get_parameters, set_parameters, sum_grad, warmup_scheduler


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
        self.eden = []
        if self.config.quantization_bit != 0:
            self.eden = eden_builder(bits=self.config.quantization_bit)
        self.model = torch.load(self.pt_path).to(config.server_device)
        self.writer = writer
        self.structure = []
        self.shape_2d = []
        self.low_rank_U = []
        self.low_rank_V = []
        self.low_rank_p_value = []

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_parameters(self.model)
        for arr in ndarrays:
            size = arr.shape
            self.structure.append(size)
            shape = size
            if len(size) > 2:
                m = 1
                for i in size[1:]:
                    m = m * i
                shape = (size[0], m)
            elif len(size) == 1:
                shape = (1, size[0])
            self.shape_2d.append(shape)
        if self.config.low_rank < 1:
            for s in self.shape_2d:
                p = min(s[0], s[1]) * self.config.low_rank
                if int(p) == 0:
                    p = 1
                self.low_rank_p_value.append(int(p))

        parameters = ndarrays_to_sparse_parameters(ndarrays)
        return parameters

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
        base_lr = self.config.client_lr
        lr = warmup_scheduler(base_lr, server_round, self.config.num_rounds)
        # Create custom configs
        standard_config = {
            "client_lr": lr,  # lr
            "server_round": server_round,
            "local_epochs": self.config.local_epoch,
            "device": self.config.device,
            "sparse_dense": self.config.sparse_dense,
            "structure": self.structure,
            "model_name": self.config.model_name,
            "low_rank_p_value": self.low_rank_p_value,
            "shape_2d": self.shape_2d,
            "quantization_bit": self.config.quantization_bit,
        }

        fit_configurations = []
        for client in clients:
            fit_configurations.append((client, FitIns(parameters, standard_config)))
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        length = len(results)
        result = []
        parameters = []
        sparse_matrixes = []
        # default
        if (
            self.config.sparse_dense == 1
            and self.config.low_rank == 1
            and self.config.quantization_bit == 0
        ):
            weights_results = [
                (
                    sparse_parameters_to_ndarrays(fit_res.parameters, self.structure),
                    fit_res.num_examples,
                )
                for _, fit_res in results
            ]
            result = aggregate(weights_results)
        # sparse O  양자화 X
        elif self.config.sparse_dense != 1 and self.config.quantization_bit == 0:
            for _, fit_res in results:
                ndarray = sparse_parameters_to_ndarrays(
                    fit_res.parameters, self.structure
                )
                params, sparse = self.sim_random_mask(ndarray)
                parameters.append(params)
                sparse_matrixes.append(sparse)

            parameters_aggregated = self.client_matrices_aggregate(parameters)
            sparse_aggregated = self.client_matrices_aggregate(sparse_matrixes)

            for p, s in zip(parameters_aggregated, sparse_aggregated):
                s[s == 0] = 1
                update = p / s
                result.append(update.to("cpu").numpy())

        elif self.config.quantization_bit != 0:

            for _, fit_res in results:
                ndarray = sparse_parameters_to_ndarrays(
                    fit_res.parameters, self.structure
                )
                params, sparse = self.sim_quantization(ndarray, self.eden)
                parameters.append(params)
                sparse_matrixes.append(sparse)

            parameters_aggregated = self.client_matrices_aggregate(parameters)
            sparse_aggregated = self.client_matrices_aggregate(sparse_matrixes)

            for p, s in zip(parameters_aggregated, sparse_aggregated):
                s[s == 0] = 1
                update = p / s
                result.append(update.to("cpu").numpy())

        # low rank
        elif self.config.low_rank != 1:

            for _, fit_res in results:
                ndarrays = sparse_parameters_to_ndarrays(
                    fit_res.parameters, self.structure
                )
                parameters.append(self.sim_low_rank_svd(ndarrays))

            tensor_result = self.client_matrices_aggregate(parameters)
            for i in range(len(tensor_result)):
                tensor_result[i] = tensor_result[i] / length
            result = [arr.to("cpu").numpy() for arr in tensor_result]
        metrics_aggregated = {}
        param = ndarrays_to_sparse_parameters(
            sum_grad(self.model, result, self.config.server_lr)
        )

        return param, metrics_aggregated

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

        parameters = sparse_parameters_to_ndarrays(parameters, self.structure)

        set_parameters(
            self.model, parameters
        )  # Update model with the latest parameters
        loss, accuracy = test(
            self.model, test_loader, self.config.server_device, self.config.model_name
        )
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

    def client_matrices_aggregate(self, ndarray_list):

        weighted_weights = [[layer for layer in weights] for weights in ndarray_list]
        weights = [
            reduce(torch.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]
        return weights

    def sim_random_mask(self, ndarray_list):
        ndarray_l = []
        sparse_l = []
        length = len(ndarray_list)
        index = 0
        for arr, shape in zip(ndarray_list, self.shape_2d):
            size = arr.shape
            arr = arr.reshape(shape)

            sparse_matrix = torch.tensor(
                scipy.sparse.random(
                    shape[0],
                    shape[1],
                    density=1 if index in [0, 1, 16, 17] else self.config.sparse_dense,
                    format="csr",
                    dtype=None,
                    random_state=np.random.randint(99999999),
                    data_rvs=np.ones,
                ).toarray()
            ).to(self.config.device)
            arr = torch.tensor(arr).to(self.config.device)
            tensor = arr.mul(sparse_matrix)
            if len(size) != 2:
                tensor = tensor.view(size)
                sparse_matrix = sparse_matrix.view(size)
            ndarray_l.append(tensor)
            sparse_l.append(sparse_matrix)
            index = index + 1
        return ndarray_l, sparse_l

    def sim_low_rank(self, ndarrays, random_state):
        ndarray_l = []
        i = 0
        for arr, size, p, seed, origin_size in zip(
            ndarrays, self.shape_2d, self.low_rank_p_value, random_state, self.structure
        ):
            if i in [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]:
                ndarray_l.append(arr.detach())
                i = i + 1
                continue
            u = torch.randn(
                (size[0], p),
                generator=torch.Generator().manual_seed(seed),
                requires_grad=False,
            ).to("cuda")
            w = torch.mm(u, arr.detach()).view(origin_size)
            ndarray_l.append(w)
            i = i + 1
        return ndarray_l

    def sim_low_rank2(self, ndarrays, random_state):
        ndarray_l = []
        i = 0
        for arr, size, p, seed, origin_size in zip(
            ndarrays, self.shape_2d, self.low_rank_p_value, random_state, self.structure
        ):
            if i in [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]:
                ndarray_l.append(arr)
                i = i + 1
                continue
            u = torch.randn(
                (size[0], p),
                generator=torch.Generator().manual_seed(seed),
                requires_grad=False,
            ).to("cuda")
            w = torch.mm(u, arr).view(origin_size)
            ndarray_l.append(w)
            i = i + 1
        return ndarray_l

    def sim_low_rank_svd(self, ndarrays):
        tensors = []
        i = 0
        for arr, origin_size, size, p in zip(
            ndarrays, self.structure, self.shape_2d, self.low_rank_p_value
        ):
            if i in [0, 1, 16, 17]:
                tensors.append(torch.tensor(arr))
                i = i + 1
                continue
            tensor = torch.tensor(arr).to("cuda")
            tensor = tensor.view(size)
            U, s, V = torch.linalg.svd(tensor)
            s = torch.diag(s[:p]).to("cuda")
            tensor = torch.mm(torch.mm(U[:, :p], s), V[:p, :])
            tensors.append(tensor.view(origin_size))
            i = i + 1
        return tensors

    def sim_quantization(self, ndarray_list, eden):
        ndarray_l = []
        sparse_l = []
        length = len(ndarray_list)
        index = 0
        for arr, shape in zip(ndarray_list, self.shape_2d):
            size = arr.shape
            arr = arr.reshape(shape)
            ndarray_zero_list = np.where(arr == 0)
            sparse_matrix = scipy.sparse.random(
                shape[0],
                shape[1],
                density=1 if index in [0, 1, 16, 17] else self.config.sparse_dense,
                format="csr",
                dtype=None,
                random_state=np.random.randint(99999999),
                data_rvs=np.ones,
            )
            indices = sparse_matrix.indices
            indptr = sparse_matrix.indptr

            sparse_matrix = torch.tensor(sparse_matrix.toarray()).to(self.config.device)
            arr = torch.tensor(arr).to(self.config.device)
            tensor = arr.mul(sparse_matrix)
            ndarray = tensor.to_sparse_csr()

            values = ndarray.values().to("cpu").numpy()

            zero_list = [[], []]
            for r, c in zip(ndarray_zero_list[0], ndarray_zero_list[1]):
                if sparse_matrix[r, c] == 1:
                    zero_list[0].append(r)
                    zero_list[1].append(c)

            # insert original zero value

            for row, col in zip(zero_list[0], zero_list[1]):
                count = 0

                for i in indices[indptr[row] :]:
                    if col == i:
                        values = np.insert(values, indptr[row] + count, 0)
                        break
                    else:
                        count = count + 1
            values = torch.tensor(values, dtype=torch.float32).to("cuda")

            encoded_x, context = eden.forward(values)
            reconstructed_x, metrics = eden.backward(encoded_x, context)
            ndarray = csr_array(
                (reconstructed_x.to("cpu").numpy(), indices, indptr), shape=shape
            ).toarray()
            tensor = torch.tensor(ndarray).to("cuda")
            if len(size) != 2:
                tensor = tensor.view(size)
                sparse_matrix = sparse_matrix.view(size)
            ndarray_l.append(tensor)
            sparse_l.append(sparse_matrix)
            index = index + 1
        return ndarray_l, sparse_l
