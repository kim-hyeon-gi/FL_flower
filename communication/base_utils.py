from io import BytesIO
from typing import cast

import numpy as np
import scipy
import torch
from flwr.common.typing import NDArray, NDArrays, Parameters


def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
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

    if len(ndarray.shape) > 2:
        ndarray = ndarray.reshape(-1, ndarray.shape[-1])
    elif len(ndarray.shape) == 1:
        ndarray = ndarray.reshape(-1, 1)

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

    return bytes_io.getvalue()


def sparse_bytes_to_ndarray(tensor: bytes, size) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    loader = np.load(bytes_io, allow_pickle=False)  # type: ignore

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

    if len(size) != 2:
        ndarray_deserialized = ndarray_deserialized.reshape(size)

    return cast(NDArray, ndarray_deserialized)
