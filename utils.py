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


def ndarrays_to_sparse_parameters(
    ndarrays: NDArrays, structure, sparse_dense, random_state=[]
) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    if sparse_dense == 1:
        random_state = [1 for i in range(len(structure))]

    tensors = []
    for ndarray, size, random in zip(ndarrays, structure, random_state):
        tensors.append(ndarray_to_sparse_bytes(ndarray, size, sparse_dense, random))
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray"), random_state


def sparse_parameters_to_ndarrays(
    parameters: Parameters, structure, sparse_dense, random_state=[], s_matrix=False
) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    if sparse_dense == 1:
        random_state = [1 for i in range(len(structure))]
    else:
        random_state = np.random.rand(len(structure))

    return [
        sparse_bytes_to_ndarray(tensor, size, sparse_dense, random, s_matrix)
        for tensor, size, random in zip(parameters.tensors, structure, random_state)
    ]


def ndarray_to_sparse_bytes(ndarray: NDArray, size, sparse_dense, random) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()

    if len(ndarray.shape) > 2:
        ndarray = ndarray.reshape(-1, ndarray.shape[-1])
    elif len(ndarray.shape) == 1:
        ndarray = ndarray.reshape(-1, 1)
    # We convert our ndarray into a sparse matrix

    zero_list = np.where(ndarray == 0)
    sparse_matrix = scipy.sparse.random(
        ndarray.shape[0],
        ndarray.shape[1],
        density=sparse_dense,
        format="csr",
        dtype=None,
        random_state=random,
        data_rvs=np.ones,
    )
    ndarray = ndarray * sparse_matrix.toarray()
    ndarray = torch.tensor(ndarray).to_sparse_csr()
    values = ndarray.values().numpy()

    # insert original zero value

    for row, col in zip(zero_list[0], zero_list[1]):
        count = 0

        for i in sparse_matrix.indices[sparse_matrix.indptr[row] :]:
            if col == i:
                values = np.insert(values, sparse_matrix.indptr[row] + count, 0)
                break
            else:
                count = count + 1

    # And send it by utilizng the sparse matrix attributes
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(
        bytes_io,  # type: ignore
        values,
        allow_pickle=False,
    )

    return bytes_io.getvalue()


def sparse_bytes_to_ndarray(
    tensor: bytes, size, sparse_dense, random, s_matrix
) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    loader = np.load(bytes_io, allow_pickle=False)  # type: ignore
    shape = size
    if len(size) > 2:
        m = 1
        for i in size[:-1]:
            m = m * i
        shape = (m, size[-1])
    elif len(size) == 1:
        shape = (size[0], 1)

    sparse_matrix = scipy.sparse.random(
        shape[0],
        shape[1],
        density=sparse_dense,
        format="csr",
        dtype=None,
        random_state=random,
        data_rvs=np.ones,
    )
    # We convert our sparse matrix back to a ndarray, using the attributes we sent
    ndarray_deserialized = csr_array(
        (loader, sparse_matrix.indices, sparse_matrix.indptr), shape=shape
    ).toarray()

    if len(size) != 2:
        ndarray_deserialized = ndarray_deserialized.reshape(size)

    if s_matrix:
        return (cast(NDArray, ndarray_deserialized), sparse_matrix.toarray())
    else:
        return cast(NDArray, ndarray_deserialized)
