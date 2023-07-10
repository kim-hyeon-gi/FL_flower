import json
from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np
import torch


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
