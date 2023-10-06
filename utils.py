import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import scipy
import torch
from flwr.common.typing import NDArray, NDArrays, Parameters
from scipy.sparse import csr_array

Scalar = Union[bool, bytes, float, int, str]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    random_state: List
    V_value: List
    metrics: Dict[str, Scalar]


class Config:
    def __init__(self, json_path):
        with open(json_path, mode="r") as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode="w") as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode="r") as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)

    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    net.load_state_dict(state_dict, strict=True)


def get_grad(net) -> List[np.ndarray]:
    return [p.grad for p in net.parameters]


def sum_grad(net, grad, lr):
    param_list = get_parameters(net)
    param = []
    for origin, grad in zip(param_list, grad):
        param.append(origin + lr * grad)

    return param


def warmup_scheduler(base_lr, round, total_rounds):

    warmup_end_round = total_rounds // 10
    if round < warmup_end_round:
        return base_lr * (round / warmup_end_round)
    elif warmup_end_round < round and round < warmup_end_round * 5:
        return base_lr
    else:
        return base_lr * (
            (total_rounds - round) / (total_rounds - 5 * warmup_end_round)
        )


def config_validation(config):
    if config.sparse_dense < 1 and config.low_rank < 1:
        raise Exception("sparse_dense and row rank can't be under the 1 at the same")
