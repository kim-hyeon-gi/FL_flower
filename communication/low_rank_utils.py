from io import BytesIO
from typing import cast

import numpy as np
import scipy
import torch
from flwr.common.typing import NDArray, NDArrays, Parameters
from numpy import linalg
from scipy.sparse import csr_array


def low_rank_ndarrays_to_sparse_parameters(
    ndarrays: NDArrays, shape_2d, low_rank_values
) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""

    tensors = []
    for ndarray, size, low_rank_value in zip(ndarrays, shape_2d, low_rank_values):
        tensors.append(low_rank_ndarray_to_sparse_bytes(ndarray, size, low_rank_value))
    return Parameters(tensors=tensors, tensor_type="list")


def low_rank_sparse_parameters_to_ndarrays(
    parameters: Parameters, structure
) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""

    return [
        low_rank_sparse_bytes_to_ndarray(tensor, size)
        for tensor, size in zip(parameters.tensors, structure)
    ]


def low_rank_ndarray_to_sparse_bytes(ndarray: NDArray, size, low_rank_value) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    svd_list = []
    if len(ndarray.shape) == 1:
        return ndarray
    if len(ndarray.shape) == 4:
        ndarray = ndarray.reshape(size)
    U, s, Vt = linalg.svd(ndarray)
    U = U[:, :low_rank_value]
    s = s[:low_rank_value]
    Vt = Vt[:low_rank_value, :]
    svd_list.append(U)
    svd_list.append(s)
    svd_list.append(Vt)

    return svd_list


def low_rank_sparse_bytes_to_ndarray(list, size) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""

    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray = []
    if len(list) == 3:
        U, s, Vt = list[0], list[1], list[2]
        Sigma = np.zeros((len(s), len(s)))
        np.fill_diagonal(Sigma, s)
        ndarray = U @ Sigma @ Vt
        ndarray = ndarray.reshape(size)

    else:
        ndarray = list

    return ndarray
