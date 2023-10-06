from io import BytesIO
from typing import cast

import numpy as np
import scipy
import torch
from flwr.common.typing import NDArray, NDArrays, Parameters
from scipy.sparse import csr_array


def efficient_communication_ndarrays_to_sparse_parameters(
    ndarrays: NDArrays, structure, sparse_dense, random_state=[]
) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    if sparse_dense == 1:
        random_state = [1 for i in range(len(structure))]
    else:
        random_state = [np.random.randint(1000000) for i in range(len(structure))]
    tensors = []
    for ndarray, size, random in zip(ndarrays, structure, random_state):
        tensors.append(
            efficient_communication_ndarray_to_sparse_bytes(
                ndarray, size, sparse_dense, random
            )
        )
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray"), random_state


def efficient_communication_sparse_parameters_to_ndarrays(
    parameters: Parameters, structure, sparse_dense, random_state
) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""

    return [
        efficient_communication_sparse_bytes_to_ndarray(
            tensor, size, sparse_dense, random
        )
        for tensor, size, random in zip(parameters.tensors, structure, random_state)
    ]


def efficient_communication_ndarray_to_sparse_bytes(
    ndarray: NDArray, size, sparse_dense, random
) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()

    if len(ndarray.shape) > 2:
        ndarray = ndarray.reshape(-1, ndarray.shape[-1])
    elif len(ndarray.shape) == 1:
        ndarray = ndarray.reshape(-1, 1)
    # We convert our ndarray into a sparse matrix

    sparse_matrix = scipy.sparse.random(
        ndarray.shape[0],
        ndarray.shape[1],
        density=sparse_dense,
        format="csr",
        dtype=None,
        random_state=random,
        data_rvs=np.ones,
    )
    ndarray_zero_list = np.where(ndarray == 0)
    ndarray = ndarray * sparse_matrix.toarray()
    ndarray = torch.tensor(ndarray).to_sparse_csr()
    values = ndarray.values().numpy()
    zero_list = [[], []]
    for r, c in zip(ndarray_zero_list[0], ndarray_zero_list[1]):
        if sparse_matrix.toarray()[r, c] == 1:
            zero_list[0].append(r)
            zero_list[1].append(c)

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


def efficient_communication_sparse_bytes_to_ndarray(
    tensor: bytes, size, sparse_dense, random
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
        shape = (1, size[0])
    # print("shape :", shape)

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
    # print(len(loader), len(sparse_matrix.indices), sparse_matrix.indptr[-1], shape)
    ndarray_deserialized = csr_array(
        (loader, sparse_matrix.indices, sparse_matrix.indptr), shape=shape
    ).toarray()
    sparse_matrix = sparse_matrix.toarray()
    if len(size) != 2:
        ndarray_deserialized = ndarray_deserialized.reshape(size)
        sparse_matrix = sparse_matrix.reshape(size)
        # print("reshape size :", np.shape(ndarray_deserialized))

    return (ndarray_deserialized, sparse_matrix.toarray())
