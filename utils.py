import json
from collections import OrderedDict
from io import BytesIO
from typing import Callable, Dict, List, cast

import numpy as np
import torch
from flwr.common.typing import NDArray, NDArrays, Parameters


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


def LSTM_set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    print(state_dict.keys())
    net.load_state_dict(state_dict, strict=True)


def get_grad(net) -> List[np.ndarray]:
    return [p.grad for p in net.parameters]


def sum_grad(net, grad):
    param_list = get_parameters(net)
    grad_list = grad
    param = []
    for origin, grad in zip(param_list, grad_list):
        param.append(origin + grad)

    return param


def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = []
    for ndarray in ndarrays:
        tensors.append(ndarray_to_sparse_bytes(ndarray))
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def client_ndarrays_to_sparse_parameters(
    ndarrays: NDArrays, sparse_dense=1
) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_sparse_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def sparse_parameters_to_ndarrays(parameters: Parameters, structure) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [
        sparse_bytes_to_ndarray(tensor, size)
        for tensor, size in zip(parameters.tensors, structure)
    ]


def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    if len(ndarray.shape) > 1:
        if len(ndarray.shape) > 2:
            ndarray = ndarray.reshape(-1, ndarray.shape[-1])
        # We convert our ndarray into a sparse matrix

        ndarray = torch.tensor(ndarray).to_sparse_csr()

        # And send it by utilizng the sparse matrix attributes
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.savez(
            bytes_io,  # type: ignore
            crow_indices=ndarray.crow_indices(),
            col_indices=ndarray.col_indices(),
            values=ndarray.values(),
            allow_pickle=False,
        )
    else:
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def sparse_bytes_to_ndarray(tensor: bytes, size) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    loader = np.load(bytes_io, allow_pickle=False)  # type: ignore

    if "crow_indices" in loader:
        # We convert our sparse matrix back to a ndarray, using the attributes we sent

        ndarray_deserialized = (
            torch.sparse_csr_tensor(
                crow_indices=loader["crow_indices"],
                col_indices=loader["col_indices"],
                values=loader["values"],
            )
            .to_dense()
            .numpy()
        )
        if len(size) > 2:
            ndarray_deserialized = ndarray_deserialized.reshape(size)
    else:
        ndarray_deserialized = loader
    return cast(NDArray, ndarray_deserialized)
